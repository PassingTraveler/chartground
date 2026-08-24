$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$src = "assets\project1\data\sft\sft_v4_combined.jsonl"
$out = "data/processed/text_math_sft.jsonl"
python -m data.build_text_math_sft --input $src --output $out --limit 100000
Write-Output "TEXT_MATH_SFT_READY: $out"
