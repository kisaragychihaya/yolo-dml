# yolo_dml

[![CI](https://github.com/kisaragychihaya/yolo-dml/actions/workflows/ci.yml/badge.svg)](https://github.com/kisaragychihaya/yolo-dml/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/kisaragychihaya/yolo-dml?label=release)](https://github.com/kisaragychihaya/yolo-dml/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Rust 编写的 YOLO DLL：OpenCV 做图像预处理（letterbox），ONNX Runtime **DirectML**（GPU 加速，自动回退 CPU）做推理，导出 C ABI，任意语言可绑定。支持**目标检测**与**姿态估计（pose）**两种任务，算法流程与 ultralytics Python 实现逐行对齐，并用官方参考输出做了数值级验证。

## 特性

- 支持目标检测（detect）与姿态估计（pose）两类任务，按模型 metadata 的 `task` / `kpt_shape` 自动判定
- 支持三类导出布局（detect / pose 各对应两种单输出布局）：
  - **end2end**：输出 `(1, N, 6)`（模型已内置 NMS，如 `yolo26n.onnx`；ultralytics `yolo export nms=True` 也是此格式）；pose 版为 `(1, N, 6+K*D)`
  - **标准格式**：输出 `(1, 4+nc, anchors)`（DLL 内部完成置信度过滤 + 类感知 NMS，兼容 YOLOv8/v11/v26 常规导出）；pose 版为 `(1, 4+nc+K*D, anchors)`
  - **torchvision 四输出**：图内带 `NonMaxSuppression` 算子的导出（`num_detections` / `detection_boxes` / `detection_scores` / `detection_classes` 四个输出），DLL 内部完成解码与坐标反算（仅 detect）
- pose 关键点（`(x, y)` 或 `(x, y, visibility)`，即 `kpt_shape` 的 `D=2/3`）与检测框走同一套几何反算与裁剪，NMS 与 detect 完全相同（关键点随候选一起被保留/抑制）
- NMS 为 SoA + 行扫描实现（无新依赖，IoU 是 broadcast+elementwise 而非 GEMM，矩阵库无收益）：候选 N=5000 时约 2.2ms（朴素 AoS 3.5ms），语义与 ultralytics 贪心 NMS 逐位一致（144 组随机配置对照验证，见 `cargo run --release --example bench_nms`）
- LetterBox 缩放/114 灰边/中心对齐、BGR→RGB、CHW、÷255，与 ultralytics `LetterBox` 完全一致
- 坐标反算采用 `scale_boxes` 的 X/Y 独立增益（gain_x/gain_y），任意宽高比无亚像素误差；关键点与检测框共用同一映射
- 类名自动读取模型 metadata 的 `names` 字段，缺省时回退内置 COCO80
- DirectML EP 初始化失败自动回退 CPU，并在 `meta.device` 中注明原因

## 目录结构

```
├── Cargo.toml / build.rs / .cargo/config.toml   # Rust 工程（cdylib + rlib）
├── src/                     # lib.rs(FFI) detector.rs preprocess.rs postprocess.rs error.rs
├── thirdparty/
│   ├── onnxruntime-dml/     # onnxruntime.dll 1.20.1 + DirectML.dll 1.15.2（NuGet 解包）
│   ├── opencv/              # OpenCV 4.10.0 官方 Windows 预编译包（仅保留 build/{include,x64/vc16}）
│   └── clang-stub/          # clang.exe 垫片（build 机专用，换机器需用 rustc 重生成，见文件头注释）
├── assets/                  # 测试图、黄金参考 JSON/叠图、dll 输出
├── examples/
│   ├── demo.py              # ctypes 全链路演示（--pose 演示姿态估计）
│   └── compare_reference.py # DLL 输出 vs 黄金参考 对比工具（含关键点比较）
├── py/                      # uv venv（onnxruntime/opencv-python/libclang），生成参考用
├── yolo26n.onnx             # 测试模型（end2end，COCO80）
├── yolo26n-pose.onnx        # 测试模型（pose，标准布局，COCO 17 关键点 x3）
└── ultralytics/             # 官方源码（只读参考，未改动）
```

## 构建

环境要求：Windows 10+、Visual Studio 2022（C++ 工具链）、Rust stable。

```bash
cargo build            # debug → target/debug/yolo_dml.dll
cargo build --release  # release → target/release/yolo_dml.dll（推理更快，建议交付用）
cargo test             # 43 个测试（含真实模型的 detect / pose 黄金门集成测试）
```

`build.rs` 会自动把 `onnxruntime.dll`、`DirectML.dll`、`opencv_world4100.dll` 等 7 个运行时 DLL 拷到 `target/<profile>/`。

> 构建机专用说明：`opencv` crate 绑定生成需要 libclang，由 `py/.venv` 里的 libclang wheel 提供（`LIBCLANG_PATH` 已写进 `.cargo/config.toml`）；`opencv-binding-generator` 另要求一个 `clang` 可执行文件，本机用 `thirdparty/clang-stub/clang.exe` 垫片代替（它只应答 include 路径查询，真正的解析由 libclang 完成）。如果机器装过 LLVM 可删除垫片与 `CLANG_PATH`。

## 部署

把以下文件放在一起即可（调用方 exe 同目录，或已加入 DLL 搜索路径）：

| 文件 | 来源 |
|---|---|
| `yolo_dml.dll` | `cargo build --release` 产物 |
| `onnxruntime.dll` | `thirdparty/onnxruntime-dml/` |
| `DirectML.dll` | `thirdparty/onnxruntime-dml/` |
| `opencv_world4100.dll` | `thirdparty/opencv/build/x64/vc16/bin/` |

运行要求：DirectML 需要支持 DX12 的 GPU（无 GPU 时 DirectML 会尝试 WARP 软件渲染，仍失败则自动回退 CPU EP）；VC++ 2015-2022 运行库（装过 VS 的机器自带）。

## C API

```c
const char* yolo_version(void);
int64_t     yolo_create(const char* model_path, const char* options_json);
char*       yolo_detect_file(int64_t handle, const char* image_path, float conf, float iou);
char*       yolo_detect_bgr(int64_t handle, const uint8_t* data, int32_t width, int32_t height, float conf, float iou);
char*       yolo_detect_bgr_strided(int64_t handle, const uint8_t* data, int32_t width, int32_t height,
                                    int32_t row_stride_bytes, float conf, float iou);

typedef struct YoloTensorView {
    const char*    name;   /* 可空；四输出布局用它匹配角色 */
    const uint8_t* data;   /* 张量原始字节，行主序 */
    const int64_t* dims;   /* ndim 个维度，batch 在前 */
    int32_t        ndim;
    int32_t        dtype;  /* 0 = float32, 1 = int32, 2 = int64 */
} YoloTensorView;

char*       yolo_decode_tensors(int64_t handle, const YoloTensorView* tensors, int32_t count,
                               int32_t src_width, int32_t src_height, float conf, float iou);
void        yolo_free_string(char* s);
void        yolo_destroy(int64_t handle);
const char* yolo_last_error(void);
```

- 字符串均为 UTF-8。`yolo_detect_*` / `yolo_decode_tensors` 返回的 JSON 用 `yolo_free_string` 释放；失败返回 NULL，错误原因用 `yolo_last_error()` 读取（线程局部，下次同线程调用前有效）。
- `options_json`：可传 NULL/空串（默认 `{"device":"dml"}`），或 `{"device":"dml"|"cpu","num_threads":n}`。
- `conf <= 0` 时用默认 0.25；`iou <= 0` 时用默认 0.7（end2end 模型的 NMS 已在模型内完成，`iou` 仅对标准格式生效）。
- `yolo_detect_bgr` 的 `data` 为紧密排列的 BGR24（stride = width*3），调用方保留所有权。
- `yolo_detect_bgr_strided` 用于带行距的 BGR24（OpenCV `Mat` 等）：`row_stride_bytes >= width*3`，`data` 至少 `(height-1)*row_stride_bytes + width*3` 字节，行间填充会被跳过。
- `yolo_decode_tensors`：推理在调用方完成，DLL 只做解码 + 置信度过滤 + NMS。语义与 `yolo_detect_*` 相同，但几何映射由 `src_width`/`src_height` 与模型输入尺寸用与 letterbox 相同的算法重建——**调用方预处理必须与本库 letterbox 一致**（等比缩放至 `input_width x input_height`、114 灰边居中、BGR→RGB、÷255）。单输出模型传 1 个张量（按形状路由，含 pose）；torchvision 四输出传 4 个张量（先按名字匹配，名字为空时按形状兜底：元素数 1 是 `num_detections`、3D 末维 4 是 `boxes`、剩下两个按 dtype 区分 `scores`(float) / `classes`(int)，同为 float 时报错）。`meta.inference_ms` 固定为 0.0。
- 句柄全局唯一，可多Detector并存；每个句柄内部有互斥锁，可跨线程调用。

返回 JSON schema（detect）：

```json
{
  "count": 5,
  "detections": [
    {"x1": 5.576, "y1": 228.866, "x2": 807.233, "y2": 749.235,
     "confidence": 0.925, "class_id": 5, "class_name": "bus"}
  ],
  "meta": {"width": 810, "height": 1080, "device": "dml", "inference_ms": 26.3, "task": "detect"}
}
```

pose 模型（`meta.task = "pose"`）每个 detection 多一个 `keypoints` 字段：`(K, 3)` 为 `[x, y, v]`（v 是图内已 sigmoid 的可见度），`(K, 2)` 为 `[x, y]`：

```json
{
  "count": 1,
  "detections": [
    {"x1": 224.5, "y1": 402.32, "x2": 345.16, "y2": 866.62,
     "confidence": 0.858, "class_id": 0, "class_name": "person",
     "keypoints": [[294.9, 448.84, 0.9974], [300.9, 440.28, 0.9794]]}
  ],
  "meta": {"width": 810, "height": 1080, "device": "dml", "inference_ms": 26.3, "task": "pose"}
}
```

坐标均为原图像素（左上角原点），裁剪到图像边界（与 ultralytics `clip_boxes` 一致，上界为 width/height）；关键点使用与检测框完全相同的 pad/gain 反算与裁剪。`meta.task` 对 detect 模型也会输出（`"detect"`），是向后兼容的新字段。

## 验证

```bash
py/.venv/Scripts/python.exe examples/demo.py   # 需先 pip install opencv-python（venv 已有）
# 输出 assets/dll_bus.json / dll_zidane.json + 叠图，并与黄金参考对比：

py/.venv/Scripts/python.exe examples/compare_reference.py assets/dll_bus.json assets/reference_bus.json
py/.venv/Scripts/python.exe examples/compare_reference.py assets/dll_zidane.json assets/reference_zidane.json

# 姿态估计：关键点同样参与对比（默认 1.5px / 0.02 可见度容差）
py/.venv/Scripts/python.exe examples/demo.py --pose
py/.venv/Scripts/python.exe examples/compare_reference.py assets/dll_pose_bus.json assets/reference_pose_bus.json
py/.venv/Scripts/python.exe examples/compare_reference.py assets/dll_pose_zidane.json assets/reference_pose_zidane.json
```

当前结果：detect 两图全 PASS（IoU 0.993~1.0，置信度差 ≤0.0003）；pose 两图同样 PASS（关键点最大偏差 <1px，可见度差 <0.01）。黄金参考由 `py/gen_reference.py` 独立生成（onnxruntime+opencv+numpy，同样严格对齐 ultralytics 算法）。

## 已知限制

- 支持 detect 与 pose 两类任务；静态 batch 必须为 1（动态 batch 按 1 处理）
- pose 模型的关键点形状优先取 metadata `kpt_shape`；缺 metadata 时只有 COCO person 两种形状能识别——`(1, 56, A)`（标准，K=17/D=3/nc=1）与 `(1, N, 57)`（end2end）。其他 K/D 组合必须保留 `kpt_shape` 元数据
- torchvision 四输出布局**只支持 detect**；pose 模型若导出成四输出，`yolo_create` 直接报错而不是静默丢掉关键点
- 输入张量限 float32 NCHW 3 通道（YOLO 标准输入）
- **图内已带 NMS 的模型**（end2end `(1,N,6)` 与 torchvision 四输出）：`iou` 参数完全无效（NMS 已按导出时阈值在图内完成），`conf` 只能往严格调
- 图内含 `NonMaxSuppression` 算子的模型在 DirectML 下会话创建成功，但该算子及其尾巴会**静默回退 CPU 执行**（ORT 分区行为），`meta.device` 仍显示 `dml`；对纯 CPU 机器无影响
- 标准格式输出的 NMS 未实现 ultralytics 的 `max_nms=30000` 上限裁剪（8400 anchors 远低于此，无实际影响）
- 上下文类名解析按 `,` 分割，类名本身含逗号会被切错（ultralytics 官方模型无此情况）

## 版本锁定

ort 2.0.0-rc.9 / ort-sys 2.0.0-rc.9 ↔ ONNX Runtime 1.20.1（API v20，混用更高版本会运行期崩溃）；opencv crate 0.99.1 ↔ OpenCV 4.10.0。升级任意一方前先核对 ABI 对应关系。

## 从源码构建

仓库只跟踪源码与构建脚本，原生依赖由脚本下载（不入 git，也不占仓库体积）。

环境要求：Windows 10/11 x64、Visual Studio 2022（"使用 C++ 的桌面开发"）、Rust stable。

```powershell
# 1) 下载并布置原生依赖：onnxruntime.dll + DirectML.dll、OpenCV 4.10.0、clang 垫片
powershell -ExecutionPolicy Bypass -File scripts/setup-deps.ps1

# 2) 构建（release 产物：target/release/yolo_dml.dll）
cargo build --release
```

`scripts/setup-deps.ps1` 幂等（已就位的依赖直接 skip），任一步失败即非零退出，`-Force` 可强制重下。它把 onnxruntime.dll / DirectML.dll 放进 `thirdparty/onnxruntime-dml/`，把 OpenCV 拍平到 `thirdparty/opencv/build/{include,x64/vc16/{bin,lib}}`，并在缺失时用 `rustc` 编译 `thirdparty/clang-stub/clang.exe`（绑定生成用的 clang 垫片，见该文件头注释）。

## 运行测试与演示

```powershell
# 下载测试图（ultralytics/assets v0.0.0）并导出 yolo26n.onnx / yolo26n-pose.onnx
# （首次会装 CPU 版 torch + ultralytics）
powershell -ExecutionPolicy Bypass -File scripts/fetch-test-assets.ps1

cargo test                                               # 43 个测试
py/.venv/Scripts/python.exe examples/demo.py --release    # ctypes 全链路演示（detect）
py/.venv/Scripts/python.exe examples/demo.py --release --pose   # 姿态估计演示（画关键点）
py/.venv/Scripts/python.exe examples/compare_reference.py assets/dll_bus.json assets/reference_bus.json
```

`scripts/fetch-test-assets.ps1` 同样幂等：图片/模型已存在就跳过，torch / ultralytics 已装就跳过。`assets/reference_*.json` 是随仓库提交的黄金参考，可用 `py/gen_reference.py` 重新生成（确定性，产物与提交的一致）。pose 集成测试（`pose_bus_matches_python_reference`）在 `yolo26n-pose.onnx` 或 `assets/reference_pose_bus.json` 缺失时自动跳过，不影响其他测试。

## 分发说明与第三方许可

- 模型权重（`yolo26n.onnx` / `yolo26n.pt`，AGPL-3.0）与 ultralytics 测试图**不随本仓库分发**，由上面的脚本在本地获取；`thirdparty/` 下的二进制同样不入库。
- 本仓库代码为 MIT（见 `LICENSE`）。release 包内再分发的第三方运行时均为可再分发组件：
  - ONNX Runtime 1.20.1 — MIT（`Microsoft.ML.OnnxRuntime.DirectML` NuGet）
  - DirectML 1.15.2 — 随 `Microsoft.AI.DirectML` NuGet 再分发
  - OpenCV 4.10.0 — Apache-2.0（并含 BSD-3-Clause 的第三方组件）

使用本项目训练的模型与权重时，请遵守其各自的许可。
