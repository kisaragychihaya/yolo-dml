<#
.SYNOPSIS
    Download and lay out the native build/runtime dependencies of yolo_dml.

.DESCRIPTION
    Populates thirdparty/ with everything build.rs and the opencv crate need:

      thirdparty/onnxruntime-dml/  onnxruntime.dll + DirectML.dll      (NuGet packages)
      thirdparty/opencv/           build/{include,x64/vc16/{bin,lib}}  (official Windows pack)
      thirdparty/clang-stub/       clang.exe                           (compiled from clang_stub.rs)

    The script is idempotent: each artifact is skipped when it is already in place,
    so it is safe to run on every build (and cheap once the CI cache is warm).

    Nothing but binaries is produced; all of it is git-ignored.

.PARAMETER Force
    Re-download / re-extract / re-compile everything, even if the artifacts exist.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup-deps.ps1

.EXAMPLE
    pwsh -NoProfile -File scripts/setup-deps.ps1 -Force
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$LASTEXITCODE = 0

# --- versions and URLs (edit here to upgrade) ------------------------------
$OnnxRuntimeVersion = '1.20.1'
$DirectMLVersion = '1.15.2'
$OpenCvVersion = '4.10.0'

$OnnxRuntimeNupkgUrl = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/$OnnxRuntimeVersion/microsoft.ml.onnxruntime.directml.$OnnxRuntimeVersion.nupkg"
$DirectMLNupkgUrl = "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/$DirectMLVersion/microsoft.ai.directml.$DirectMLVersion.nupkg"

# The official SourceForge /download endpoint throttles to a crawl; the mirror below
# serves the same 4.10.0 self-extracting 7z archive at full speed.
$OpenCvExeUrl = "https://master.dl.sourceforge.net/project/opencvlibrary/$OpenCvVersion/opencv-$OpenCvVersion-windows.exe?viasf=1"

# --- paths -----------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ThirdParty = Join-Path $RepoRoot 'thirdparty'
$DownloadDir = Join-Path $ThirdParty '.downloads'

$OrtDir = Join-Path $ThirdParty 'onnxruntime-dml'
$OpenCvDir = Join-Path $ThirdParty 'opencv'
$ClangStubDir = Join-Path $ThirdParty 'clang-stub'

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
function Invoke-NativeCommand {
    <#
        Runs a native program and judges it by its exit code.

        With $ErrorActionPreference = 'Stop', Windows PowerShell turns a single line of a
        native program's stderr into a terminating NativeCommandError, which would abort
        the script on harmless warnings (rustc, curl --show-error, tar). The preference is
        therefore relaxed for the duration of the call.
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

function New-Directory {
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Get-RemoteFile {
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][string]$Destination
    )

    New-Directory (Split-Path -Parent $Destination)
    Write-Host "  downloading $Url"
    Write-Host "           ->  $Destination"

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

    if (-not (Test-Path -LiteralPath $Destination)) {
        throw "download produced no file: $Destination"
    }
    $size = (Get-Item -LiteralPath $Destination).Length
    if ($size -le 0) {
        throw "downloaded file is empty: $Destination"
    }
    Write-Host ("  got {0:N1} MB" -f ($size / 1MB))
}

function Expand-ZipArchive {
    param(
        [Parameter(Mandatory)][string]$Archive,
        [Parameter(Mandatory)][string]$Destination
    )

    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    New-Directory $Destination

    # bsdtar (shipped with Windows 10 1803+) unpacks .nupkg (zip) archives quickly.
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (Test-Path -LiteralPath $tar) {
        Invoke-NativeCommand -What "tar -xf $Archive" -Command {
            & $tar -xf $Archive -C $Destination
        }
    }
    else {
        Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    }
}

function Find-X64DllIn {
    <#
        The NuGet packages ship the same DLL for every architecture (win-x64, win-arm64,
        win-x86, ...), so a plain recursive search can silently pick an arm64 or x86
        build that then fails to load. Filter those out, prefer x64 by path, and let
        Assert-PeX64 check the machine type for real.
    #>
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Name
    )

    $notX64 = '(?i)(arm64|aarch64|arm-win|win-arm|x86-win|win-x86|/x86/)'
    $hits = @(Get-ChildItem -LiteralPath $Root -Recurse -File -Filter $Name -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch $notX64 })
    if ($hits.Count -eq 0) {
        throw "could not find an x64 '$Name' inside $Root"
    }
    $x64Hits = @($hits | Where-Object { $_.FullName -match '(?i)x64' })
    if ($x64Hits.Count -gt 0) {
        $hits = $x64Hits
    }
    Write-Host ("  using {0}" -f $hits[0].FullName)
    return $hits[0].FullName
}

function Assert-PeX64 {
    param([Parameter(Mandatory)][string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $reader = New-Object System.IO.BinaryReader($stream)
        if ($stream.Length -lt 0x40 -or $reader.ReadUInt16() -ne 0x5A4D) {
            throw "$Path is not a PE file"
        }
        $stream.Position = 0x3C
        $peOffset = $reader.ReadInt32()
        $stream.Position = $peOffset
        if ($reader.ReadUInt32() -ne 0x00004550) {
            throw "$Path has no PE signature"
        }
        $machine = $reader.ReadUInt16()
    }
    finally {
        $stream.Dispose()
    }
    if ($machine -ne 0x8664) {
        throw ("{0} is not an x64 DLL (PE machine 0x{1:X4})" -f $Path, $machine)
    }
}

function Get-DirectorySize {
    param([Parameter(Mandatory)][string]$Path)

    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) {
        return [long]0
    }
    return [long]$sum
}

function Find-OpenCvBuildDir {
    param([Parameter(Mandatory)][string]$Root)

    return Get-ChildItem -LiteralPath $Root -Recurse -Directory -Filter 'build' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'include\opencv2\opencv.hpp') } |
        Select-Object -First 1
}

function Wait-OpenCvExtraction {
    <#
        The OpenCV Windows package is a 7-Zip self-extracting archive whose launcher hands
        the work to a detached second process and returns immediately, so the extraction
        must be waited for rather than assumed finished. Poll until the build tree exists
        and the extracted size stopped growing.
    #>
    param(
        [Parameter(Mandatory)][string]$Root,
        [int]$TimeoutMinutes = 15
    )

    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    $previousSize = -1
    $stableRounds = 0

    while ($true) {
        $size = Get-DirectorySize -Path $Root
        $buildDir = Find-OpenCvBuildDir -Root $Root
        if ($buildDir -and ($size -eq $previousSize)) {
            $stableRounds++
            if ($stableRounds -ge 2) {
                return $buildDir
            }
        }
        else {
            $stableRounds = 0
        }
        $previousSize = $size

        if ((Get-Date) -gt $deadline) {
            throw "timed out after $TimeoutMinutes min waiting for the OpenCV self-extractor (extracted so far: $size bytes under $Root)"
        }
        Start-Sleep -Seconds 3
    }
}

# ---------------------------------------------------------------------------
# 1) ONNX Runtime DirectML (onnxruntime.dll + DirectML.dll)
# ---------------------------------------------------------------------------
function Install-OnnxRuntimeDirectML {
    $ortDll = Join-Path $OrtDir 'onnxruntime.dll'
    $directMlDll = Join-Path $OrtDir 'DirectML.dll'

    if (-not $Force -and (Test-Path -LiteralPath $ortDll) -and (Test-Path -LiteralPath $directMlDll)) {
        Write-Host "[skip] onnxruntime-dml: onnxruntime.dll + DirectML.dll already present"
        return
    }

    Write-Host "[get ] onnxruntime-dml $OnnxRuntimeVersion + DirectML $DirectMLVersion"
    New-Directory $OrtDir

    $ortNupkg = Join-Path $DownloadDir "microsoft.ml.onnxruntime.directml.$OnnxRuntimeVersion.nupkg"
    if (Test-Path -LiteralPath $ortNupkg) {
        Write-Host "  reusing archive left behind by an earlier run: $ortNupkg"
    }
    else {
        Get-RemoteFile -Url $OnnxRuntimeNupkgUrl -Destination $ortNupkg
    }
    $ortExtract = Join-Path $DownloadDir 'onnxruntime-dml'
    Expand-ZipArchive -Archive $ortNupkg -Destination $ortExtract
    Copy-Item -LiteralPath (Find-X64DllIn $ortExtract 'onnxruntime.dll') -Destination $ortDll -Force

    $directMlNupkg = Join-Path $DownloadDir "microsoft.ai.directml.$DirectMLVersion.nupkg"
    if (Test-Path -LiteralPath $directMlNupkg) {
        Write-Host "  reusing archive left behind by an earlier run: $directMlNupkg"
    }
    else {
        Get-RemoteFile -Url $DirectMLNupkgUrl -Destination $directMlNupkg
    }
    $directMlExtract = Join-Path $DownloadDir 'directml'
    Expand-ZipArchive -Archive $directMlNupkg -Destination $directMlExtract
    Copy-Item -LiteralPath (Find-X64DllIn $directMlExtract 'DirectML.dll') -Destination $directMlDll -Force

    foreach ($dll in @($ortDll, $directMlDll)) {
        if (-not (Test-Path -LiteralPath $dll)) {
            throw "failed to lay out $dll"
        }
        Assert-PeX64 -Path $dll
    }
    Write-Host "[ok  ] onnxruntime-dml -> $OrtDir"
}

# ---------------------------------------------------------------------------
# 2) OpenCV prebuilt Windows package (headers + vc16 lib/bin)
# ---------------------------------------------------------------------------
function Install-OpenCv {
    $includeMarker = Join-Path $OpenCvDir 'build\include\opencv2\opencv.hpp'
    $libMarker = Join-Path $OpenCvDir 'build\x64\vc16\lib\opencv_world4100.lib'
    $binMarker = Join-Path $OpenCvDir 'build\x64\vc16\bin\opencv_world4100.dll'

    if (-not $Force -and (Test-Path -LiteralPath $includeMarker) -and
        (Test-Path -LiteralPath $libMarker) -and (Test-Path -LiteralPath $binMarker)) {
        Write-Host "[skip] opencv: build/{include,x64/vc16/{bin,lib}} already present"
        return
    }

    Write-Host "[get ] opencv $OpenCvVersion (official Windows package, ~183 MB)"
    $exe = Join-Path $DownloadDir "opencv-$OpenCvVersion-windows.exe"
    if (Test-Path -LiteralPath $exe) {
        Write-Host "  reusing archive left behind by an earlier run: $exe"
    }
    else {
        Get-RemoteFile -Url $OpenCvExeUrl -Destination $exe
    }

    # Self-extracting 7z archive: -y answers all prompts, -o<dir> must not have a space.
    $extractRoot = Join-Path $DownloadDir 'opencv-extract'
    if (Test-Path -LiteralPath $extractRoot) {
        Remove-Item -LiteralPath $extractRoot -Recurse -Force
    }
    New-Directory $extractRoot
    Write-Host "  extracting to $extractRoot (the archive detaches an extractor; waiting for it)"
    Invoke-NativeCommand -What 'opencv self-extract' -Command {
        & $exe -y "-o$extractRoot"
    }

    # The archive wraps everything in an extra opencv/ level (and, occasionally,
    # opencv/opencv/), so locate build/ by its content instead of a fixed depth.
    $buildDir = Wait-OpenCvExtraction -Root $extractRoot
    if (-not $buildDir) {
        throw "unrecognised OpenCV archive layout: no build/include/opencv2/opencv.hpp under $extractRoot"
    }

    Write-Host "  flattening $($buildDir.FullName)"
    if (Test-Path -LiteralPath $OpenCvDir) {
        Remove-Item -LiteralPath $OpenCvDir -Recurse -Force
    }
    $dstBuild = Join-Path $OpenCvDir 'build'
    New-Directory (Join-Path $dstBuild 'include')
    Copy-Item -Path (Join-Path $buildDir.FullName 'include\*') -Destination (Join-Path $dstBuild 'include') -Recurse -Force

    foreach ($sub in @('bin', 'lib')) {
        $src = Join-Path $buildDir.FullName "x64\vc16\$sub"
        if (-not (Test-Path -LiteralPath $src)) {
            throw "OpenCV archive is missing x64/vc16/$sub (expected at $src)"
        }
        $dst = Join-Path $dstBuild "x64\vc16\$sub"
        New-Directory $dst
        Copy-Item -Path (Join-Path $src '*') -Destination $dst -Recurse -Force
    }

    # Debug symbols are ~300 MB of the package and neither the linker nor the runtime
    # needs them; drop them so the checkout and the CI cache stay small.
    Get-ChildItem -LiteralPath $OpenCvDir -Recurse -File -Filter '*.pdb' -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue

    if (-not (Test-Path -LiteralPath $includeMarker) -or -not (Test-Path -LiteralPath $libMarker) -or
        -not (Test-Path -LiteralPath $binMarker)) {
        throw "OpenCV flattening incomplete; check $OpenCvDir"
    }
    Assert-PeX64 -Path $binMarker
    Write-Host "[ok  ] opencv -> $OpenCvDir"
}

# ---------------------------------------------------------------------------
# 3) clang.exe shim (see thirdparty/clang-stub/clang_stub.rs for the rationale)
# ---------------------------------------------------------------------------
function Install-ClangStub {
    $clangExe = Join-Path $ClangStubDir 'clang.exe'
    $clangSource = Join-Path $ClangStubDir 'clang_stub.rs'

    if (-not $Force -and (Test-Path -LiteralPath $clangExe)) {
        Write-Host "[skip] clang-stub: clang.exe already built"
        return
    }
    if (-not (Test-Path -LiteralPath $clangSource)) {
        throw "missing shim source: $clangSource"
    }

    $rustc = Get-Command rustc -ErrorAction SilentlyContinue
    if (-not $rustc) {
        throw "rustc not found on PATH; install Rust (https://rustup.rs) before running this script (it compiles the clang shim)"
    }

    Write-Host "[get ] clang-stub: compiling clang_stub.rs with $($rustc.Source)"
    Push-Location $ClangStubDir
    try {
        Invoke-NativeCommand -What 'rustc clang_stub.rs' -Command {
            & rustc -O -o clang.exe clang_stub.rs
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $clangExe)) {
        throw "rustc did not produce $clangExe"
    }
    Write-Host "[ok  ] clang-stub -> $clangExe"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
Write-Host "yolo_dml native dependencies"
Write-Host "  repo root : $RepoRoot"
Write-Host "  force     : $($Force.IsPresent)"

try {
    Install-OnnxRuntimeDirectML
    Install-OpenCv
    Install-ClangStub
}
finally {
    # Keep the checkout (and the CI cache) free of multi-hundred-MB archives; with
    # -Force the next run simply downloads them again.
    if (Test-Path -LiteralPath $DownloadDir) {
        Remove-Item -LiteralPath $DownloadDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "native dependencies ready:"
Write-Host "  thirdparty/onnxruntime-dml  onnxruntime.dll $OnnxRuntimeVersion, DirectML.dll $DirectMLVersion"
Write-Host "  thirdparty/opencv           OpenCV $OpenCvVersion (build/include, build/x64/vc16/{bin,lib})"
Write-Host "  thirdparty/clang-stub       clang.exe shim"
