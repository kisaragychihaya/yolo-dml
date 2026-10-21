<#
.SYNOPSIS
    Fetch the test images and the ONNX model that yolo_dml's tests and demo need.

.DESCRIPTION
    The repository does not redistribute Ultralytics' sample images or the exported
    model (see README), so this script fetches them:

      assets/bus.jpg, assets/zidane.jpg   ultralytics/assets release v0.0.0
                                          (gh CLI first, plain curl fallback)
      yolo26n.onnx                        exported locally from yolo26n.pt with the
                                          `ultralytics` package inside py/.venv

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
$ModelPath = Join-Path $RepoRoot 'yolo26n.onnx'
$ImageNames = @('bus.jpg', 'zidane.jpg')

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
        [Parameter(Mandatory)][string]$Module
    )

    $exitCode = Invoke-NativeCommand -What "python -c import $Module" -AllowFailure -Command {
        & $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$Module') else 1)"
    }
    return ($exitCode -eq 0)
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
# 2) yolo26n.onnx (exported, never committed: model weights are not redistributed)
# ---------------------------------------------------------------------------
function Install-Model {
    if (Test-Path -LiteralPath $ModelPath) {
        Write-Host "[skip] model: yolo26n.onnx already present"
        return
    }

    Write-Host "[get ] model: exporting yolo26n.onnx from yolo26n.pt"

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
    if (Test-PythonModule -Python $VenvPython -Module 'torch') {
        Write-Host "[skip] torch already installed"
    }
    else {
        Write-Host "  installing CPU torch"
        Invoke-NativeCommand -What 'uv pip install torch' -Command {
            & $uv.Source pip install --python $VenvPython torch --index-url https://download.pytorch.org/whl/cpu
        }
    }

    if (Test-PythonModule -Python $VenvPython -Module 'ultralytics') {
        Write-Host "[skip] ultralytics already installed"
    }
    else {
        # Deliberately unpinned; if a future ultralytics release changes the YOLO26 ONNX
        # export defaults (py/gen_reference.py assumes a (1, 300, 6) end2end output),
        # pin the version here instead.
        Write-Host "  installing ultralytics"
        Invoke-NativeCommand -What 'uv pip install ultralytics' -Command {
            & $uv.Source pip install --python $VenvPython ultralytics
        }
    }

    if (-not (Test-Path -LiteralPath $VenvYolo)) {
        throw "missing $VenvYolo; the ultralytics install did not provide the yolo entry point"
    }

    # ultralytics downloads yolo26n.pt and writes yolo26n.onnx next to it, i.e. into the
    # current directory - run the export from the repository root.
    Push-Location $RepoRoot
    try {
        Invoke-NativeCommand -What 'yolo export model=yolo26n.pt format=onnx' -Command {
            & $VenvYolo export model=yolo26n.pt format=onnx
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $ModelPath)) {
        throw "export finished but $ModelPath is missing"
    }
    Write-Host ("[ok  ] model -> {0} ({1:N1} MB)" -f $ModelPath, ((Get-Item -LiteralPath $ModelPath).Length / 1MB))
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
Write-Host "yolo_dml test assets"
Write-Host "  repo root : $RepoRoot"

Install-TestImages
Install-Model

Write-Host ""
Write-Host "test assets ready: assets\bus.jpg, assets\zidane.jpg, yolo26n.onnx"
