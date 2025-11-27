Write-Host "=== Creating deployment package ===" -ForegroundColor Cyan

# 1. 自动生成 .deployment 文件，确保配置正确
Write-Host "`nGenerating .deployment file..." -ForegroundColor Yellow
$deploymentContent = @"
[config]
SCM_DO_BUILD_DURING_DEPLOYMENT = true
ENABLE_ORYX_BUILD = true
"@
Set-Content -Path ".\.deployment" -Value $deploymentContent
Write-Host "   .deployment file created with Oryx build flags." -ForegroundColor Green

# 创建临时部署目录
$deployDir = ".\deploy_temp"
if (Test-Path $deployDir) { 
    Remove-Item $deployDir -Recurse -Force 
}
New-Item -ItemType Directory -Path $deployDir | Out-Null

# 要部署的文件列表
$filesToDeploy = @(
    "main.py",
    "requirements.txt",
    "startup.sh",
    "pyproject.toml",
    "README.md",
    ".deployment"
)

Write-Host "`nCopying files to deployment directory..." -ForegroundColor Yellow

foreach ($file in $filesToDeploy) {
    if (Test-Path $file) {
        Copy-Item -Path $file -Destination $deployDir
        $size = (Get-Item $file).Length
        Write-Host "   $file ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "   $file (NOT FOUND) - Please check if this file exists!" -ForegroundColor Red
    }
}

# 验证 main.py
Write-Host "`nVerifying main.py..." -ForegroundColor Yellow
if (Test-Path "$deployDir\main.py") {
    $mainPyLines = (Get-Content "$deployDir\main.py" | Measure-Object -Line).Lines
    Write-Host "   main.py: $mainPyLines lines" -ForegroundColor Green
}

# 显示部署目录内容
Write-Host "`nDeployment directory contents:" -ForegroundColor Yellow
Get-ChildItem $deployDir | Format-Table Name, Length, LastWriteTime -AutoSize

# 创建 ZIP 文件
Write-Host "`nCreating ZIP archive..." -ForegroundColor Yellow
$zipPath = ".\deploy.zip"
if (Test-Path $zipPath) { 
    Remove-Item $zipPath -Force 
}
Compress-Archive -Path "$deployDir\*" -DestinationPath $zipPath -Force

# 显示 ZIP 信息
if (Test-Path $zipPath) {
    $zipSize = (Get-Item $zipPath).Length
    Write-Host "`n ZIP file created successfully!" -ForegroundColor Green
    Write-Host "   Location: $(Resolve-Path $zipPath)" -ForegroundColor Cyan
    Write-Host "   Size: $([math]::Round($zipSize / 1KB, 2)) KB" -ForegroundColor Cyan
} else {
    Write-Host "`n ERROR: ZIP file was not created!" -ForegroundColor Red
}

# 清理临时目录
Remove-Item $deployDir -Recurse -Force
Write-Host "`n Temporary directory cleaned up" -ForegroundColor Green

Write-Host "`n=== Ready to deploy! ===" -ForegroundColor Cyan
Write-Host "Next step: Run the following command to deploy:" -ForegroundColor Yellow
Write-Host "az webapp deployment source config-zip --name <app-name> --resource-group <resource-group> --src deploy.zip" -ForegroundColor White
