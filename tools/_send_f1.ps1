param(
  [Parameter(Mandatory=$true)][string]$Script,
  [string]$Envp = "",
  [int]$Tmo = 120,
  [string[]]$Aux = @()
)
# Same wrapper as _send_ct.ps1 but aimed at the sibling node f01-1. f01-2 stopped answering ssh
# after a reboot that a wedged amdgpu shutdown appears to have hung; f01-1 is the other gfx1250
# box in the same rack and it is idle.
$k = "$env:USERPROFILE\Desktop\my\id_ed25519"
$h = "fizhang@ctheliosr-1b114-f01-1.mnb.dcgpu"
$raw = (Get-Content $Script -Raw) -replace "`r`n", "`n"
$raw = $raw.TrimStart([char]0xFEFF)
$Aux = @($Aux | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' })
if ($Aux.Count -gt 0) {
  $tmp = Join-Path $env:TEMP "ctaux"
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  $staged = @()
  foreach ($f in $Aux) {
    $n = Split-Path $f -Leaf
    $c = ((Get-Content $f -Raw) -replace "`r`n", "`n").TrimStart([char]0xFEFF)
    $p = Join-Path $tmp $n
    [IO.File]::WriteAllText($p, $c, (New-Object Text.UTF8Encoding $false))
    $staged += $p
  }
  scp -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i $k $staged "${h}:/tmp/" | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Error "scp of -Aux files failed"; exit 1 }
}
$b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($raw))
$cmd = "echo '===CTOUT==='; echo $b64 | base64 -d | $Envp timeout $Tmo bash -s"
$out = ssh -o BatchMode=yes -o ConnectTimeout=900 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=60 -i $k $h $cmd 2>&1
$i = 0
for ($j = 0; $j -lt $out.Count; $j++) { if ($out[$j] -match '===CTOUT===') { $i = $j + 1; break } }
$out[$i..($out.Count - 1)]
