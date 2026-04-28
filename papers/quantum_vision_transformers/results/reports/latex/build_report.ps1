param(
    [string]$TexFile = "qvt_scientific_analysis.tex"
)

$ErrorActionPreference = "Stop"

$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
if (-not $pdflatex) {
    $fallback = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
    if (Test-Path $fallback) {
        $pdflatex = $fallback
    } else {
        throw "pdflatex was not found on PATH and no MiKTeX fallback was found."
    }
} else {
    $pdflatex = $pdflatex.Source
}

$reportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $reportDir

function Invoke-Latex {
    param(
        [string]$WorkingDirectory,
        [string]$InputFile
    )

    Push-Location $WorkingDirectory
    try {
        & $pdflatex -interaction=nonstopmode -halt-on-error $InputFile | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "pdflatex failed for $InputFile"
        }
    } finally {
        Pop-Location
    }
}

if (Test-Path (Join-Path $reportDir "figures")) {
    Get-ChildItem -Path (Join-Path $reportDir "figures") -Filter *.tex | Sort-Object Name | ForEach-Object {
        Invoke-Latex -WorkingDirectory $_.DirectoryName -InputFile $_.Name
        Invoke-Latex -WorkingDirectory $_.DirectoryName -InputFile $_.Name
    }
}

Invoke-Latex -WorkingDirectory $reportDir -InputFile $TexFile
Invoke-Latex -WorkingDirectory $reportDir -InputFile $TexFile
