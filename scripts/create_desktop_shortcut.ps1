$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$root = 'e:\Work Space\Git Repositories\DeepDive'
$lnk = $ws.CreateShortcut((Join-Path $desktop 'Delveta.lnk'))
$lnk.TargetPath = (Join-Path $root 'scripts\launch-delveta.vbs')
$lnk.WorkingDirectory = $root
$lnk.IconLocation = (Join-Path $root 'scripts\delveta-electron.ico')
$lnk.Description = 'Delveta desktop workbench launcher'
$lnk.Save()
Write-Output "Created: $(Join-Path $desktop 'Delveta.lnk')"
