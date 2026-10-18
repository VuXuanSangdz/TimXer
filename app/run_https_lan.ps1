<#
Chạy app qua HTTPS trong mạng LAN để điện thoại/laptop khác GHI ÂM được.
(Trình duyệt chỉ cho phép micro trên https:// hoặc localhost — mở http://192.168.x.x:7860 sẽ bị chặn micro.)

Dùng:  .\app\run_https_lan.ps1            # tạo chứng chỉ tự ký (lần đầu) rồi chạy uvicorn --ssl trên cổng 7860
Sau đó trên điện thoại (cùng Wi-Fi) mở  https://<IP máy>:7860/collect  -> chấp nhận cảnh báo chứng chỉ (Advanced -> Proceed).
#>
param([int]$Port = 7860)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$certDir = Join-Path $Root "app\certs"; New-Item -ItemType Directory -Force $certDir | Out-Null
$key = Join-Path $certDir "dev-key.pem"; $crt = Join-Path $certDir "dev-cert.pem"

$ips = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*" } | Select-Object -ExpandProperty IPAddress
if (-not (Test-Path $crt)) {
    $openssl = (Get-Command openssl -ErrorAction SilentlyContinue).Source
    if (-not $openssl -and (Test-Path "C:\Program Files\Git\usr\bin\openssl.exe")) { $openssl = "C:\Program Files\Git\usr\bin\openssl.exe" }
    if (-not $openssl) { Write-Error "Không tìm thấy openssl. Cài Git for Windows hoặc 'pip install trustme' rồi tạo cert thủ công." }
    $san = (@("DNS:localhost", "IP:127.0.0.1") + ($ips | ForEach-Object { "IP:$_" })) -join ","
    & $openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes -keyout $key -out $crt -subj "/CN=TimeXer AI dev" -addext "subjectAltName=$san" 2>$null
    Write-Host "Đã tạo chứng chỉ tự ký: $crt (SAN: $san)"
}
Write-Host "Mở trên điện thoại cùng Wi-Fi:"; $ips | ForEach-Object { Write-Host "  https://$_`:$Port/collect" }
$env:PYTHONIOENCODING = "utf-8"
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port $Port --ssl-keyfile $key --ssl-certfile $crt
