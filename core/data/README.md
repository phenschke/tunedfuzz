# Pretrained cost tables

`Costs::pretrained(name)` returns one of these directed tables. Each edit `(a, b)` replaces `a` in the truth with `b` in the reading.

| Name | Learned from |
|---|---|
| `general` | Every training corpus below, each weighted to the same number of words |
| `antiqua` | The corpora in Antiqua (roman) type: Overproof and ICDAR DE4, EN1, ES1, FR1, FR2, FR3 and SL1 |
| `fraktur` | The corpora in Fraktur: the Munich address books and ICDAR CZ1, DE1, DE2, DE3 and DE7 |

Each ICDAR subset's type was judged from its pages and its misreadings, because OCR of Fraktur turns k into f and V into B. The Kurrent registers, Bulgarian (Cyrillic) and the subsets in older blackletter or of unclear type (DE5, DE6, NL1, PL1) train only `general`.

## Training corpora

| Corpus | Text | Engine | Words | Licence |
|---|---|---|---|---|
| ICDAR 2019 post-OCR, 17 subsets | Printed books and newspapers, 15th–20th c., BG, CZ, DE, EN, ES, FR, NL, PL, SL and Latin | Various | 2.6M | CC-BY 4.0 |
| Overproof dataset 1 | Sydney Morning Herald, 1842–1954 | ABBYY FineReader | 8.1M | Not stated |
| Munich address books | Surnames in Fraktur directories, 1845–1957 | Gemini | 6.2M | Private, not published |
| Munich civil registers | Kurrent birth and death certificates, 1876–1882 | Transkribus Titan I and II | 55k | Private, not published |

## Test corpora

| Corpus | Text | Engine | Words | Licence |
|---|---|---|---|---|
| BLN600 | London newspapers, 19th c. | Not stated | 259k | CC BY-NC 4.0 |
| Overproof datasets 2 and 3 | Sydney Morning Herald and Chronicling America | ABBYY FineReader and others | 64k | Not stated |
| IMPACT-BHL | Natural-history books, 1708–1913, EN, FR, DE and Latin | Tesseract 5, Kraken, 19 VLMs | 9.4M | CC-BY 3.0, outputs per model |
| RISE benchmark | Basler Avisblatt adverts in Fraktur, 1731–1762, and a 15th-c. manuscript | About 100 LLMs | 348k | CC-BY 4.0 |

## Citations

- ICDAR 2019: Rigaud, Doucet, Coustaty and Moreux (2019), ICDAR 2019 Competition on Post-OCR Text
  Correction. <https://zenodo.org/record/3515403>
- Overproof: Evershed and Fitch (2014), Correcting noisy OCR. <http://overproof.projectcomputing.com/evaluation>
- BLN600: University of Sheffield (2024). <https://github.com/Shef-AIRE/vlms_post-ocr_correction>
- IMPACT-BHL ground truth: IMPACT and the Biodiversity Heritage Library.
  <https://huggingface.co/datasets/finebooks/bhl-impact-gt>. Outputs from the BHL OCR leaderboard,
  <https://huggingface.co/buckets/finebooks/bhl-ocr-outputs>
- RISE humanities data benchmark, University of Basel. <https://doi.org/10.5281/zenodo.23821532>

## How the tables are built

[pretrained/](../../pretrained/README.md) holds the scripts.

1. Split each aligned text into word pairs of truth and reading. Unaligned sources pair words inside
   equal-length blocks of a word diff.
2. Keep pairs whose truth is all letters. Normalize the truth with NFKC.
3. Weight each corpus to 100,000 words, so the large corpora don't outweigh the small ones. Equal
   weights scored better in a leave-one-corpus-out test than raw counts or square-root weights.
4. Leave out the test corpora.
5. Learn a directed table with `learn_costs(symmetric=False)` and doubled letters, rn/m, cl/d,
   li/h, vv/w, ck/k and tz/z as seeds.
6. Learn `antiqua` and `fraktur` the same way from the corpora in that type.
