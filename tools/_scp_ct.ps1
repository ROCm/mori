param([Parameter(Mandatory=$true)][string[]]$Pairs, [string]$Node = "f01-2")
# Pairs: "localpath=remotename" (remote is relative to ~). scp prints the login banner per file, so
# its output is dropped entirely and only a one-line LF-normalised md5 per file is reported.
$k = "$env:USERPROFILE\Desktop\my\id_ed25519"
$h = "fizhang@ctheliosr-1b114-$Node.mnb.dcgpu"
foreach ($p in $Pairs) {
  $l, $r = $p -split '=', 2
  # ConnectTimeout must match _send_ct.ps1: the Conductor authorization step can take several hundred
  # seconds and the default (~21s) reports a bare "Connection timed out" that looks like a dead node.
  scp -o BatchMode=yes -o ConnectTimeout=900 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=60 -i $k $l "${h}:~/$r" *>$null
  if ($LASTEXITCODE -ne 0) { Write-Output "SCP FAIL $l"; continue }
  $s = [System.Text.Encoding]::UTF8.GetString([System.IO.File]::ReadAllBytes($l)) -replace "`r`n", "`n"
  $hh = [System.Security.Cryptography.MD5]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($s))
  Write-Output ("{0}  {1}" -f ((($hh | ForEach-Object { $_.ToString('x2') }) -join '').Substring(0, 12)), $r)
}
