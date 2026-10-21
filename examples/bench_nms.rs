//! Benchmark of the production NMS against the reference implementation.
//!
//!     cargo run --release --example bench_nms
//!
//! Candidate sets mimic what a YOLO decode hands to NMS: overlapping hypotheses per object,
//! a handful of classes and uniformly distributed confidences. Numbers are only meaningful in a
//! `--release` build.

use std::hint::black_box;
use std::time::{Duration, Instant};

use yolo_dml::postprocess::{nms, nms_reference, Candidate};

/// Candidate counts and iteration counts: the O(n^2) reference is much slower, so the larger
/// sets get fewer repetitions.
const CASES: [(usize, u32); 3] = [(200, 300), (1000, 120), (5000, 40)];

const IOU_THRESHOLD: f32 = 0.7;
const MAX_DET: usize = 300;

/// Deterministic `xorshift64*` PRNG, keeps the example dependency free.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    fn below(&mut self, bound: usize) -> usize {
        self.next_u32() as usize % bound
    }
}

/// `count` candidates: every candidate belongs to one of `count / 24` synthetic objects and
/// shares that object's class and (roughly) its size, with a uniform confidence. That is what a
/// real decode hands to NMS — several overlapping hypotheses per object — so both the
/// comparison and the suppression path do real work.
fn synthetic_candidates(seed: u64, count: usize) -> Vec<Candidate> {
    let mut rng = Rng::new(seed);
    // (centre x, centre y, width, height, class) per object.
    let objects: Vec<(f32, f32, f32, f32, i32)> = (0..((count + 23) / 24).max(2))
        .map(|_| {
            (
                rng.next_f32() * 600.0 + 20.0,
                rng.next_f32() * 600.0 + 20.0,
                rng.next_f32() * 160.0 + 40.0,
                rng.next_f32() * 160.0 + 40.0,
                rng.below(5) as i32,
            )
        })
        .collect();
    (0..count)
        .map(|_| {
            let (cx, cy, w, h, class_id) = objects[rng.below(objects.len())];
            // same object, slightly different box: heavy overlap, same class
            let scale = 0.85 + rng.next_f32() * 0.3;
            let (w, h) = (w * scale, h * scale);
            let cx = cx + (rng.next_f32() - 0.5) * w * 0.15;
            let cy = cy + (rng.next_f32() - 0.5) * h * 0.15;
            Candidate {
                x1: cx - w / 2.0,
                y1: cy - h / 2.0,
                x2: cx + w / 2.0,
                y2: cy + h / 2.0,
                confidence: rng.next_f32(),
                class_id,
            }
        })
        .collect()
}

/// One implementation's timings: total and best over the recorded runs, plus its last kept count.
#[derive(Default)]
struct Sample {
    total: Duration,
    best: Option<Duration>,
    kept: usize,
}

impl Sample {
    fn record(&mut self, elapsed: Duration, kept: usize) {
        self.total += elapsed;
        self.best = Some(self.best.map_or(elapsed, |best| best.min(elapsed)));
        self.kept = kept;
    }

    fn mean(&self, iterations: u32) -> Duration {
        self.total / iterations
    }

    fn best(&self) -> Duration {
        self.best.unwrap_or_default()
    }
}

/// Time both implementations in the same loop, so that CPU frequency drift and background load
/// hit them equally.
fn time_both(candidates: &[Candidate], iterations: u32) -> (Sample, Sample) {
    let mut fast = Sample::default();
    let mut reference = Sample::default();
    for _ in 0..iterations {
        let started = Instant::now();
        let kept = nms(black_box(candidates), IOU_THRESHOLD, MAX_DET).len();
        fast.record(started.elapsed(), kept);

        let started = Instant::now();
        let kept = nms_reference(black_box(candidates), IOU_THRESHOLD, MAX_DET).len();
        reference.record(started.elapsed(), kept);
    }
    (fast, reference)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn main() {
    println!(
        "{:>6}  {:<10} {:>10} {:>10} {:>13} {:>6}",
        "N", "impl", "mean ms", "min ms", "per pair ns", "kept"
    );
    for (count, iterations) in CASES {
        let candidates = synthetic_candidates(0xC0FFEE, count);
        let pairs = (count * (count - 1) / 2).max(1) as f64;
        // not timed: the two implementations must agree, otherwise the numbers mean nothing
        assert_eq!(
            nms(&candidates, IOU_THRESHOLD, MAX_DET),
            nms_reference(&candidates, IOU_THRESHOLD, MAX_DET),
            "the two implementations must agree on N={count}"
        );

        let (fast, reference) = time_both(&candidates, iterations);
        for (label, sample) in [("nms", &fast), ("reference", &reference)] {
            println!(
                "{count:>6}  {label:<10} {:>10.3} {:>10.3} {:>13.2} {:>6}",
                millis(sample.mean(iterations)),
                millis(sample.best()),
                millis(sample.best()) * 1e6 / pairs,
                sample.kept
            );
        }
        println!(
            "{:>6}  {:<10} {:>10.2}",
            "",
            "speedup",
            millis(reference.mean(iterations)) / millis(fast.mean(iterations))
        );
    }
}
