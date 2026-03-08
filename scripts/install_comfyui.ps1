<#
.SYNOPSIS
    Downloads and installs the latest ComfyUI portable package from GitHub.

.DESCRIPTION
    Fetches the latest release from comfyanonymous/ComfyUI, downloads the
    portable NVIDIA package, and extracts it to the target directory.
    The portable package includes embedded Python, PyTorch, and CUDA.

.PARAMETER TargetDir
    Directory to install ComfyUI into. Defaults to the parent of this script's
    repository (i.e., alongside the looper folder).

.PARAMETER Force
    Overwrite an existing ComfyUI installation if present.

.EXAMPLE
    .\install_comfyui.ps1
    .\install_comfyui.ps1 -TargetDir "D:\AI\ComfyUI" -Force
#>

param(
    [string]$TargetDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Default target: parent of the repo root (two levels up from scripts/)
if (-not $TargetDir) {
    $ScriptDir = Split-Path -Parent $PSCommandPath
    $RepoRoot = Split-Path -Parent $ScriptDir
    $TargetDir = Split-Path -Parent $RepoRoot
}

$ComfyUIDir = Join-Path $TargetDir "ComfyUI_windows_portable"

# Check for existing installation
if ((Test-Path $ComfyUIDir) -and -not $Force) {
    Write-Host "ComfyUI already exists at: $ComfyUIDir"
    Write-Host "Use -Force to overwrite."
    exit 0
}

# Query GitHub for latest release
Write-Host "Fetching latest ComfyUI release info..."
$ReleaseUrl = "https://api.github.com/repos/comfyanonymous/ComfyUI/releases/latest"
$Release = Invoke-RestMethod -Uri $ReleaseUrl -Headers @{ "User-Agent" = "ComfyUI-Looper-Installer" }

Write-Host "Latest release: $($Release.tag_name) - $($Release.name)"

# Find the portable NVIDIA asset
$Asset = $Release.assets | Where-Object { $_.name -match "windows.*portable.*nvidia" -or $_.name -match "ComfyUI_windows_portable.*\.7z" } | Select-Object -First 1

if (-not $Asset) {
    # Fallback: look for any .7z or .zip asset
    $Asset = $Release.assets | Where-Object { $_.name -match "\.(7z|zip)$" } | Select-Object -First 1
}

if (-not $Asset) {
    Write-Error "Could not find a portable package in the latest release. Available assets:"
    $Release.assets | ForEach-Object { Write-Host "  - $($_.name)" }
    exit 1
}

$DownloadUrl = $Asset.browser_download_url
$FileName = $Asset.name
$DownloadPath = Join-Path $env:TEMP $FileName

Write-Host "Downloading: $FileName ($([math]::Round($Asset.size / 1MB, 1)) MB)..."
Write-Host "  From: $DownloadUrl"

# Download with progress
$ProgressPreference = 'SilentlyContinue'  # Speeds up Invoke-WebRequest significantly
Invoke-WebRequest -Uri $DownloadUrl -OutFile $DownloadPath
$ProgressPreference = 'Continue'

Write-Host "Download complete: $DownloadPath"

# Extract
Write-Host "Extracting to: $TargetDir"

if ($FileName -match "\.7z$") {
    # Try 7-Zip
    $SevenZip = Get-Command "7z" -ErrorAction SilentlyContinue
    if (-not $SevenZip) {
        $SevenZipPath = "C:\Program Files\7-Zip\7z.exe"
        if (Test-Path $SevenZipPath) {
            $SevenZip = Get-Command $SevenZipPath
        }
    }

    if ($SevenZip) {
        & $SevenZip.Source x $DownloadPath -o"$TargetDir" -y
    } else {
        Write-Error @"
7-Zip is required to extract .7z files but was not found.
Install it from: https://www.7-zip.org/
Or install via winget: winget install 7zip.7zip
"@
        exit 1
    }
} elseif ($FileName -match "\.zip$") {
    Expand-Archive -Path $DownloadPath -DestinationPath $TargetDir -Force
} else {
    Write-Error "Unknown archive format: $FileName"
    exit 1
}

# Verify
$MainPy = Join-Path $ComfyUIDir "ComfyUI" "main.py"
if (-not (Test-Path $MainPy)) {
    # Check if extracted into a different folder name
    $ExtractedDirs = Get-ChildItem -Path $TargetDir -Directory | Where-Object { $_.Name -match "ComfyUI" }
    if ($ExtractedDirs) {
        $ActualDir = $ExtractedDirs[0].FullName
        $MainPy = Join-Path $ActualDir "ComfyUI" "main.py"
        if (-not (Test-Path $MainPy)) {
            $MainPy = Join-Path $ActualDir "main.py"
        }
    }
}

if (Test-Path $MainPy) {
    Write-Host ""
    Write-Host "ComfyUI installed successfully!"
    Write-Host "  Location: $ComfyUIDir"
    Write-Host ""
    Write-Host "To start ComfyUI (accessible from WSL/network):"
    Write-Host "  cd $ComfyUIDir"
    Write-Host "  .\python_embeded\python.exe -s ComfyUI\main.py --listen 0.0.0.0"
} else {
    Write-Warning "Extraction completed but could not verify installation."
    Write-Warning "Check $TargetDir for the extracted files."
}

# Cleanup
Remove-Item $DownloadPath -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "Temporary download cleaned up."
