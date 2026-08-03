param(
  [Parameter(Mandatory=$true)][string]$Script,
  [string]$Envp = "",
  [int]$Tmo = 120,
  [string[]]$Aux = @(),
  # Which reserved node to talk to. Both are the same image; f01-2 stays the default because every
  # existing caller assumes it, but reservations get recycled one node at a time.
  [string]$Node = "f01-2"
)
# The node prints a ~40-line Conductor banner on every login shell. Filtering it line-by-line let
# the ASCII art through, so instead the remote side emits a sentinel and we keep only what follows.
$k = "$env:USERPROFILE\Desktop\my\id_ed25519"
$h = "fizhang@ctheliosr-1b114-$Node.mnb.dcgpu"
$raw = (Get-Content $Script -Raw) -replace "`r`n", "`n"
$raw = $raw.TrimStart([char]0xFEFF)
# -Aux drops files into the node's /tmp before the script body runs. Without this a script can only
# reach sources that were hand-staged on the node earlier, which silently goes stale against edits here.
# These go over scp rather than inline base64: embedding them made the ssh command line exceed the
# Windows 32k limit once the payload grew past a few hundred lines.
# Under `powershell -File`, a comma-separated list arrives as one string rather than an array, so
# split again here to accept both forms.
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
# "Timeout, server not responding" was OUR client giving up, not the node dying: ServerAliveInterval=30
# with the default ServerAliveCountMax=3 disconnects after 90s of silence, and the Conductor
# authorization step now takes several hundred seconds (scp, which has no such limit, went through on
# the same key in ~390s). CountMax=60 tolerates 30 minutes of silence.
$out = ssh -o BatchMode=yes -o ConnectTimeout=900 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=60 -i $k $h $cmd 2>&1
$i = 0
for ($j = 0; $j -lt $out.Count; $j++) { if ($out[$j] -match '===CTOUT===') { $i = $j + 1; break } }
$out[$i..($out.Count - 1)]
