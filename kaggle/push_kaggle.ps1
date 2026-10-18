<#
Đẩy (1) dataset public và (2) notebook "Quy trình thu thập & xử lý dữ liệu" lên Kaggle.

Yêu cầu một lần:
  pip install kaggle
  Tải API token tại https://www.kaggle.com/settings  ->  lưu vào  %USERPROFILE%\.kaggle\kaggle.json

Dùng:
  .\kaggle\push_kaggle.ps1                 # dataset + notebook
  .\kaggle\push_kaggle.ps1 -SkipDataset    # chỉ push notebook (dataset đã có)
  .\kaggle\push_kaggle.ps1 -NoAudio        # dataset không kèm FLAC (nhẹ hơn, nhưng notebook cần audio ở mục 2/5)
#>
param(
    [string]$Username = "sangvuxuan",
    [string]$DatasetSlug = "voice-age-timexer-dinov3-vi",
    [switch]$SkipDataset,
    [switch]$NoAudio
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "$env:USERPROFILE\.kaggle\kaggle.json") -and -not $env:KAGGLE_KEY) {
    Write-Error "Chưa có kaggle.json. Tải tại https://www.kaggle.com/settings -> %USERPROFILE%\.kaggle\kaggle.json"
}

# ---------- (1) Dataset ----------
if (-not $SkipDataset) {
    $pkgArgs = @("scripts/06_package_kaggle.py", "--username", $Username, "--slug", $DatasetSlug)
    if (-not $NoAudio) { $pkgArgs += "--with-audio" }
    python @pkgArgs

    $exists = (kaggle datasets list -m -s $DatasetSlug 2>$null | Select-String "$Username/$DatasetSlug")
    if ($exists) {
        Write-Host ">> Dataset đã tồn tại -> tạo version mới"
        kaggle datasets version -p data/kaggle_dataset -m "update $(Get-Date -Format s)" --dir-mode zip
    } else {
        Write-Host ">> Tạo dataset public mới"
        kaggle datasets create -p data/kaggle_dataset --dir-mode zip --public
    }
    Write-Host "Dataset: https://www.kaggle.com/datasets/$Username/$DatasetSlug"
    Write-Host "(Kaggle cần vài phút để xử lý dataset trước khi notebook có thể đọc)"
}

# ---------- (2) Notebook ----------
python kaggle/make_data_pipeline_notebook.py $DatasetSlug
$metaPath = "kaggle/data_pipeline/kernel-metadata.json"
$meta = Get-Content $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
$meta.dataset_sources = @("$Username/$DatasetSlug")
if ($meta.id -notmatch "^$Username/") { $meta.id = "$Username/" + ($meta.id -split "/")[-1] }
$meta | ConvertTo-Json -Depth 5 | Set-Content $metaPath -Encoding UTF8
kaggle kernels push -p kaggle/data_pipeline
Write-Host "Notebook: https://www.kaggle.com/code/$($meta.id)"
Write-Host "Theo dõi trạng thái chạy: kaggle kernels status $($meta.id)"
