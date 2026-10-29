<#
.SYNOPSIS
    Fetch the test images and the ONNX models that yolo_dml's tests and demo need.

.DESCRIPTION
    The repository does not redistribute Ultralytics' sample images or the exported
    models (see README), so this script fetches them:

      assets/bus.jpg, assets/zidane.jpg,  ultralytics/assets release v0.0.0
      assets/boats.jpg                    (gh CLI first, plain curl fallback);
                                          boats.jpg is the aerial image the OBB goldens use
      yolo26n.onnx                        exported locally from yolo26n.pt with the
                                          `ultralytics` package inside py/.venv
      yolo26n-pose.onnx                   same, from yolo26n-pose.pt (pose model; the
                                          detect tests do not need it, the pose tests
                                          and `examples/demo.py --pose` do)
      yolo26n-obb.onnx                    OBB model, end-to-end (1, N, 7) layout
      yolo11n-obb.onnx                    OBB model, standard (1, 4+nc+1, anchors) layout

    Idempotent: existing files are skipped, and torch / ultralytics are only installed
    when the venv does not already provide them.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/fetch-test-assets.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$LASTEXITCODE = 0

# --- where the assets live (edit here when upstream re-tags) ----------------
$AssetsRepo = 'ultralytics/assets'
# bus.jpg / zidane.jpg are only published in the initial asset release; the newest
# release (v8.4.0) carries model weights but not these sample images.
$AssetsTag = 'v0.0.0'

# --- paths -----------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
$AssetsDir = Join-Path $RepoRoot 'assets'
$VenvDir = Join-Path $RepoRoot 'py\.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$VenvYolo = Join-Path $VenvDir 'Scripts\yolo.exe'
$ImageNames = @('bus.jpg', 'zidane.jpg', 'boats.jpg')
# Model names, exported as "<name>.onnx" from "<name>.pt" into the repository root.
# yolo26n-obb covers the end-to-end OBB layout, yolo11n-obb the standard one.
$ModelNames = @('yolo26n', 'yolo26n-pose', 'yolo26n-obb', 'yolo11n-obb')

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
function Invoke-NativeCommand {
    <#
        Runs a native program and judges it by its exit code.

        With $ErrorActionPreference = 'Stop', Windows PowerShell turns a single line of a
        native program's stderr into a terminating NativeCommandError, which would abort
        the script on harmless output (curl --show-error, gh, ultralytics warnings). The
        preference is therefore relaxed for the duration of the call.
    #>
    param(
        [Parameter(Mandatory)][string]$What,
        [Parameter(Mandatory)][scriptblock]$Command,
        [switch]$AllowFailure
    )

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command | Out-Host
    }
    finally {
        $ErrorActionPreference = $previous
    }

    if ($LASTEXITCODE -ne 0) {
        if ($AllowFailure) {
            return $LASTEXITCODE
        }
        throw "$What failed with exit code $LASTEXITCODE"
    }
    # Only report the exit code to callers that asked for it, so the ordinary
    # invocation does not leave stray "0" lines in the log.
    if ($AllowFailure) {
        return 0
    }
}

function Get-RemoteFile {
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][string]$Destination
    )

    Write-Host "  downloading $Url"
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (Test-Path -LiteralPath $curl) {
        Invoke-NativeCommand -What "curl $Url" -Command {
            & $curl -L --fail --retry 3 --retry-delay 5 --silent --show-error -o $Destination $Url
        }
    }
    else {
        $previousProgress = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
        }
        finally {
            $ProgressPreference = $previousProgress
        }
    }

    if (-not (Test-Path -LiteralPath $Destination) -or (Get-Item -LiteralPath $Destination).Length -le 0) {
        throw "download produced no usable file: $Destination"
    }
}

function Test-PythonModule {
    param(
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string]$Module,
        [Parameter(Mandatory)][string]$VenvDir
    )

    # `-I` (isolated mode) keeps the caller's cwd off sys.path: the working tree ships
    # a top-level directory named `ultralytics` (the read-only reference clone) that
    # Python would otherwise resolve as a namespace package and report as "installed".
    # The reported origin is additionally required to live inside the venv, so a hit
    # can never come from the system interpreter or a stale exit code.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $origin = & $Python -I -c "import importlib.util; s = importlib.util.find_spec('$Module'); print(s.origin if s and s.origin else '')" 2>$null
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0 -or -not $origin) {
        return $false
    }
    $venv = ($VenvDir -replace '/', '\').TrimEnd('\')
    $inside = ($origin -replace '/', '\').StartsWith($venv + '\', [System.StringComparison]::OrdinalIgnoreCase)
    Write-Host "  [check] $Module -> $origin (inside venv: $inside)"
    return $inside
}

# ---------------------------------------------------------------------------
# 1) sample images
# ---------------------------------------------------------------------------
function Install-TestImages {
    if (-not (Test-Path -LiteralPath $AssetsDir)) {
        New-Item -ItemType Directory -Force -Path $AssetsDir | Out-Null
    }

    $missing = @($ImageNames | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $AssetsDir $_))
    })
    if ($missing.Count -eq 0) {
        Write-Host "[skip] test images: $($ImageNames -join ', ') already in assets\"
        return
    }

    Write-Host "[get ] test images: $($missing -join ', ')"
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if ($gh) {
        $ghArgs = @('release', 'download', $AssetsTag, '--repo', $AssetsRepo, '--dir', $AssetsDir)
        foreach ($name in $missing) {
            $ghArgs += @('--pattern', $name)
        }
        $ghExit = Invoke-NativeCommand -What 'gh release download' -AllowFailure -Command {
            & $gh.Source @ghArgs
        }
        if ($ghExit -ne 0) {
            Write-Warning "gh release download failed (exit $ghExit); falling back to curl"
        }
    }
    else {
        Write-Host "  gh CLI not found; using curl"
    }

    $stillMissing = @($missing | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $AssetsDir $_))
    })
    foreach ($name in $stillMissing) {
        $url = "https://github.com/$AssetsRepo/releases/download/$AssetsTag/$name"
        Get-RemoteFile -Url $url -Destination (Join-Path $AssetsDir $name)
    }

    foreach ($name in $ImageNames) {
        $path = Join-Path $AssetsDir $name
        if (-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -le 0) {
            throw "failed to fetch test image $name"
        }
    }
    Write-Host "[ok  ] test images -> $AssetsDir"
}

# ---------------------------------------------------------------------------
# 2) yolo26n.onnx / yolo26n-pose.onnx (exported, never committed: model weights are
#    not redistributed)
# ---------------------------------------------------------------------------
function Initialize-PythonEnv {
    <#
        Prepare py/.venv with CPU torch + ultralytics. Only called when a model has to be
        exported, and every step skips work that is already done.
    #>
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "uv not found on PATH; install uv (https://docs.astral.sh/uv/) so py/.venv can be prepared"
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host "  creating py\.venv"
        # --seed: also install pip, so `python -m pip ...` keeps working for the
        # packages ultralytics auto-installs during export and for the user.
        Invoke-NativeCommand -What 'uv venv py\.venv' -Command {
            & $uv.Source venv --seed $VenvDir
        }
        if (-not (Test-Path -LiteralPath $VenvPython)) {
            throw "py\.venv was not created: $VenvPython is missing"
        }
    }

    # CPU wheels first: the default PyPI `torch` drags in ~2.5 GB of CUDA libraries
    # that nothing here uses.
    if (Test-PythonModule -Python $VenvPython -Module 'torch' -VenvDir $VenvDir) {
        Write-Host "[skip] torch already installed"
    }
    else {
        Write-Host "  installing CPU torch"
        Invoke-NativeCommand -What 'uv pip install torch' -Command {
            & $uv.Source pip install --python $VenvPython torch --index-url https://download.pytorch.org/whl/cpu
        }
    }

    if (Test-PythonModule -Python $VenvPython -Module 'ultralytics' -VenvDir $VenvDir) {
        Write-Host "[skip] ultralytics already installed"
    }
    else {
        # Deliberately unpinned; if a future ultralytics release changes the YOLO26 ONNX
        # export defaults (py/gen_reference.py assumes a (1, 300, 6) end2end output and
        # the pose export a (4+nc+K*D) / (6+K*D) one), pin the version here instead.
        Write-Host "  installing ultralytics"
        Invoke-NativeCommand -What 'uv pip install ultralytics' -Command {
            & $uv.Source pip install --python $VenvPython ultralytics
        }
    }

    if (-not (Test-Path -LiteralPath $VenvYolo)) {
        throw "missing $VenvYolo; the ultralytics install did not provide the yolo entry point"
    }
}

function Export-Model {
    param([Parameter(Mandatory)][string]$Name)

    $modelPath = Join-Path $RepoRoot "$Name.onnx"
    if (Test-Path -LiteralPath $modelPath) {
        Write-Host "[skip] model: $Name.onnx already present"
        return
    }

    Write-Host "[get ] model: exporting $Name.onnx from $Name.pt"
    # ultralytics downloads <name>.pt and writes <name>.onnx next to it, i.e. into the
    # current directory - run the export from the repository root.
    Push-Location $RepoRoot
    try {
        Invoke-NativeCommand -What "yolo export model=$Name.pt format=onnx" -Command {
            & $VenvYolo export "model=$Name.pt" format=onnx
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $modelPath)) {
        throw "export finished but $modelPath is missing"
    }
    Write-Host ("[ok  ] model -> {0} ({1:N1} MB)" -f $modelPath, ((Get-Item -LiteralPath $modelPath).Length / 1MB))
}

function Install-Models {
    $missing = @($ModelNames | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $RepoRoot "$_.onnx"))
    })
    if ($missing.Count -eq 0) {
        Write-Host "[skip] models: $($ModelNames -join ', ') already exported"
    }
    else {
        Initialize-PythonEnv
        foreach ($name in $ModelNames) {
            Export-Model -Name $name
        }
    }
    Install-ObbEnd2EndModel
}

function Install-ObbEnd2EndModel {
    <#
        yolo26n-obb-e2e.onnx: the same yolo26n-obb.pt exported with nms=True, i.e. the
        graph-baked rotated-NMS (1, N, 7) layout. The exporter writes to yolo26n-obb.onnx
        regardless, so the standard export has to be moved aside / restored around it.
    #>
    $e2ePath = Join-Path $RepoRoot 'yolo26n-obb-e2e.onnx'
    $standardPath = Join-Path $RepoRoot 'yolo26n-obb.onnx'
    if (Test-Path -LiteralPath $e2ePath) {
        Write-Host "[skip] model: yolo26n-obb-e2e.onnx already present"
        return
    }

    Initialize-PythonEnv
    Write-Host "[get ] model: exporting yolo26n-obb-e2e.onnx from yolo26n-obb.pt (nms=True)"
    Push-Location $RepoRoot
    try {
        Invoke-NativeCommand -What "yolo export model=yolo26n-obb.pt format=onnx nms=True" -Command {
            & $VenvYolo export "model=yolo26n-obb.pt" format=onnx "nms=True"
        }
        Move-Item -LiteralPath $standardPath -Destination $e2ePath -Force
        Write-Host "  re-exporting the standard yolo26n-obb.onnx (the e2e export clobbered it)"
        Invoke-NativeCommand -What "yolo export model=yolo26n-obb.pt format=onnx" -Command {
            & $VenvYolo export "model=yolo26n-obb.pt" format=onnx
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $e2ePath) -or -not (Test-Path -LiteralPath $standardPath)) {
        throw "e2e OBB export failed: yolo26n-obb-e2e.onnx / yolo26n-obb.onnx incomplete"
    }
    Write-Host "[ok  ] model -> $e2ePath"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
Write-Host "yolo_dml test assets"
Write-Host "  repo root : $RepoRoot"

Install-TestImages
Install-Models

Write-Host ""
Write-Host "test assets ready: assets\bus.jpg, assets\zidane.jpg, assets\boats.jpg, $($ModelNames -join ', ').onnx"
