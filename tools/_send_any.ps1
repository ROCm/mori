param(
  [Parameter(Mandatory=$true)][string]$Host_,
  [Parameter(Mandatory=$true)][string]$Script,
  [string]$Envp = "",
  [int]$Tmo = 120
)
# Same wrapper as _send_ct.ps1 with the node as an argument. Two gfx1250 boxes have now stopped
# answering ssh mid-run, so which node is alive changes faster than a per-node copy of this file.
$k = "$env:USERPROFILE\Desktop\my\id_ed25519"
$h = "fizhang@$Host_"
$raw = (Get-Content $Script -Raw) -replace "`r`n", "`n"
$raw = $raw.TrimStart([char]0xFEFF)
$b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($raw))
$cmd = "echo '===CTOUT==='; echo $b64 | base64 -d | $Envp timeout $Tmo bash -s"
$out = ssh -o BatchMode=yes -o ConnectTimeout=60 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=60 -i $k $h $cmd 2>&1
$i = 0
for ($j = 0; $j -lt $out.Count; $j++) { if ($out[$j] -match '===CTOUT===') { $i = $j + 1; break } }
$out[$i..($out.Count - 1)]
