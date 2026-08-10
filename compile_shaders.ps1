$VulkanSDK = if ($env:VULKAN_SDK) { $env:VULKAN_SDK } else { 'C:\VulkanSDK\1.4.357.0' };
$glslang = Join-Path $VulkanSDK 'Bin\glslangValidator.exe';
if (-not (Test-Path $glslang)) { $glslang = 'glslangValidator' };
$SpvDir = 'shaders/spv';
if (-not (Test-Path $SpvDir)) { New-Item -ItemType Directory -Path $SpvDir | Out-Null };
$AnyFailed = $false;
$shaders = Get-ChildItem -Path 'shaders' -Recurse -Filter '*.comp' | Where-Object { $_.FullName -notmatch 'spv' };
foreach ($comp in $shaders) {
    $name = $comp.BaseName;
    $tierDir = (Split-Path $comp.FullName -Parent);
    $tierName = (Split-Path $tierDir -Leaf);
    $spvFile = Join-Path $SpvDir "$tierName.$name.spv";
    $args = @('-V', '--target-env', 'vulkan1.4', '-e', 'main', '-o', $spvFile, $comp.FullName);
    $proc = Start-Process -FilePath $glslang -ArgumentList $args -Wait -NoNewWindow -PassThru -RedirectStandardError "$spvFile.err";
    if ($proc.ExitCode -ne 0) {
        $err = Get-Content "$spvFile.err" -Raw;
        Write-Error "glslangValidator failed for $($comp.FullName):$err";
        $AnyFailed = $true;
    } else {
        Write-Host "Compiled: $($comp.FullName) -> $spvFile";
    }
}
if ($AnyFailed) { exit 1; }
Write-Host 'All shaders compiled successfully';