$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$out = "data/generated/smoke"
python -m data.gen_figure_math --count 12 --out-dir "$out/images" --manifest "$out/train.jsonl" --seed 10000 --split train
python -m data.smoke_data --manifest "$out/train.jsonl" --expected-images 12
python -m data.gen_figure_math --count 8 --out-dir "$out/eval_images" --manifest "$out/eval.jsonl" --seed 20260814 --split eval
python -m data.smoke_data --manifest "$out/eval.jsonl" --expected-images 8
python -m data.clean_images --input-dir "$out/images" --output-dir "$out/clean_images" --report "$out/clean_report.json"
Write-Output "DATA_SMOKE_OK"

