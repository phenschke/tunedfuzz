#!/usr/bin/env bash
# Downloads the public corpora into $1 (default ./downloads). Needs curl, unzip, git and uv.
set -euo pipefail
dir=${1:-downloads}
mkdir -p "$dir/overproof/d1" "$dir/overproof/d2" "$dir/overproof/d3"

# ICDAR 2019 post-OCR (CC-BY 4.0)
curl -sL -o "$dir/icdar2019.zip" https://zenodo.org/api/records/3515403/files/ICDAR2019-POCR-ground-truth.zip/content
unzip -q -o "$dir/icdar2019.zip" 'full_22M_without_Finnish/*' -d "$dir"

# Overproof (licence not stated)
op=http://overproof.projectcomputing.com/datasets
for f in $(curl -sL "$op/dataset1/rawTextAndHumanCorrectionPairs/" | grep -o 'href="smh[0-9]*\.txt"' | cut -d'"' -f2); do
  curl -sL -o "$dir/overproof/d1/$f" "$op/dataset1/rawTextAndHumanCorrectionPairs/$f"
done
curl -sL -o "$dir/overproof/d2/allArticles.txt" "$op/dataset2/rawTextAndHumanCorrectionAndOverproofCorrectionTriples/allArticles.txt"
curl -sL -o "$dir/overproof/d3/allArticles.txt" "$op/dataset3/rawTextAndHumanCorrectionAndOverproofCorrectionTriples/allArticles.txt"

# BLN600 (CC BY-NC), test only
curl -sL -o "$dir/bln600.csv" https://raw.githubusercontent.com/Shef-AIRE/vlms_post-ocr_correction/main/data/dataset.csv

# IMPACT-BHL ground truth (CC-BY 3.0) and the BHL OCR leaderboard outputs, test only
hf="uvx --from huggingface_hub hf"
$hf download finebooks/bhl-impact-gt metadata.parquet --repo-type dataset --local-dir "$dir/bhl/gt"
$hf buckets sync hf://buckets/finebooks/bhl-ocr-outputs "$dir/bhl/outputs"

# RISE humanities data benchmark (CC-BY 4.0), Fraktur adverts and medieval manuscripts, test only
git clone -q --depth 1 --branch 2026-10-29.6 --filter=blob:none --sparse \
  https://github.com/RISE-UNIBAS/humanities_data_benchmark_dataset "$dir/rise"
git -C "$dir/rise" sparse-checkout set payloads/fraktur_adverts payloads/medieval_manuscripts
