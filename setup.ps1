$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    $projectLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($projectLauncher) {
        & py -3.12 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    } else {
        $projectBundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (-not (Test-Path -LiteralPath $projectBundledPython)) {
            throw 'Python 3.12 was not found. Set up Python 3.12 before continuing.'
        }
        & $projectBundledPython -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed. Use Python 3.12.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Setup complete. Next: .\.venv\Scripts\python.exe main.py run'
