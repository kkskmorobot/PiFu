$ErrorActionPreference = 'Stop'

$frontendRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $frontendRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$serverPath = Join-Path $frontendRoot 'app.py'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "未找到项目 Python 环境：$pythonPath"
}
if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
    throw "未找到网页服务：$serverPath"
}

Set-Location -LiteralPath $projectRoot
& $pythonPath $serverPath
