param(
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv'
$vendorInit = Join-Path $projectRoot 'vendor\ultralytics\__init__.py'
if (-not (Test-Path -LiteralPath $vendorInit -PathType Leaf)) {
    throw "Missing preserved framework: $vendorInit"
}
if (Test-Path -LiteralPath $venvRoot) {
    $venvItem = Get-Item -LiteralPath $venvRoot -Force
    if ($venvItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Refusing to install into a linked environment: $venvRoot. Use a local .venv directory."
    }
}

$baseArguments = @()
if ($PythonExe) {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python executable not found: $PythonExe" }
    $basePython = (Resolve-Path -LiteralPath $PythonExe).Path
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $basePython = (Get-Command py).Source
    $baseArguments = @('-3.13')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $basePython = (Get-Command python).Source
} else {
    throw 'Install 64-bit Python 3.13.7, or pass -PythonExe with its executable path.'
}

& $basePython @baseArguments -c "import sys,struct; assert sys.version_info[:2] == (3,13), 'Python 3.13 is required'; assert struct.calcsize('P') == 8, '64-bit Python is required'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed.' }
& $basePython @baseArguments -m venv $venvRoot
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'

function Invoke-VenvPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE." }
}

Invoke-VenvPython -Arguments @('-m', 'pip', 'install', 'pip==25.2', 'setuptools==80.9.0', 'wheel==0.45.1')
if ($Device -eq 'cuda') {
    Invoke-VenvPython -Arguments @('-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu129', 'torch==2.8.0+cu129', 'torchvision==0.23.0+cu129')
} else {
    Invoke-VenvPython -Arguments @('-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.8.0+cpu', 'torchvision==0.23.0+cpu')
}
Invoke-VenvPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $PSScriptRoot 'requirements.txt'))
Invoke-VenvPython -Arguments @('-m', 'pip', 'install', '--no-deps', 'ultralytics==8.4.101')
Invoke-VenvPython -Arguments @('-m', 'pip', 'check')

$probe = @'
import pathlib, sys
sys.path.insert(0, sys.argv[1])
import bootstrap
bootstrap.configure()
import torch, torchvision, ultralytics, cv2, yaml, numpy, pandas
from ultralytics.nn.modules import SemanticSegment
from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor
from ultralytics.data.dataset import SemanticDataset
actual = pathlib.Path(ultralytics.__file__).resolve()
assert actual.is_relative_to(bootstrap.VENDOR_ROOT.resolve()), actual
assert ultralytics.__version__ == '8.4.101', ultralytics.__version__
print('Framework:', actual)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if sys.argv[2] == 'cuda' and not torch.cuda.is_available():
    print('WARNING: CUDA build installed, but no usable NVIDIA GPU was detected. CPU inference remains available; GPU training requires a compatible driver/device.')
print('Reproduction environment ready.')
'@
Invoke-VenvPython -Arguments @('-B', '-c', $probe, $PSScriptRoot, $Device)
Write-Host "Installed environment: $venvPython" -ForegroundColor Green
if ($Device -eq 'cpu') {
    Write-Host 'CPU mode is for inference/functional checks; the original training and frozen metrics used CUDA.'
}
