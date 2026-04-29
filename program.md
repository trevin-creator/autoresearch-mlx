# invoice-ocr-autoresearch

This repo is configured as an autonomous experiment loop for invoice OCR and structured extraction.

## Goal

Maximize invoice extraction `accuracy` on the fixed labeled set in `../Training_invoices`.

Use `adjusted_score` as a secondary metric. It starts from accuracy and penalizes average document cost, latency, low throughput, and crashes. Do not chase speed or cost at the expense of meaningful accuracy unless accuracy is essentially tied.

## Files

- `score_invoices.py` - fixed evaluator. Do not mutate during an experiment run.
- `train.py` - mutable experiment runner. Change model/provider/prompt/parameters here.
- `results.tsv` - experiment history.
- `run.log` - ignored local run output.
- `predictions.jsonl` and `invoice_report.json` - ignored local artifacts.

## Data

The training/eval set is one folder up:

```bash
../Training_invoices
```

Each invoice should have a document file plus matching JSON label by stem, for example `1.pdf` and `1.json`.

The label JSON has header fields plus `Rows`. The scorer canonicalizes field names and formats, so equivalent names like `InvoiceId` and `Invoice No`, or `$1,000.00` and `1000`, can match.

## Credentials

Never commit secrets. The runner reads credentials only from environment variables:

```bash
export MISTRAL_API_KEY=...
export OPENROUTER_API_KEY=...
export AZURE_FORM_RECOGNIZER_ENDPOINT=...
export AZURE_FORM_RECOGNIZER_KEY=...
export HF_TOKEN=...
```

Optional cost overrides:

```bash
export MISTRAL_OCR_COST_PER_PAGE=0.001
export MISTRAL_INPUT_USD_PER_MILLION=0.15
export MISTRAL_OUTPUT_USD_PER_MILLION=0.60
export AZURE_PREBUILT_COST_PER_DOC=0.01
```

## Running

Default baseline:

```bash
uv run train.py > run.log 2>&1
```

Switch experiments with `INVOICE_EXPERIMENT`:

```bash
INVOICE_EXPERIMENT=mistral_ocr_small4_v1 uv run train.py > run.log 2>&1
INVOICE_EXPERIMENT=mistral_ocr_small4_table_html uv run train.py > run.log 2>&1
INVOICE_EXPERIMENT=azure_prebuilt_invoice uv run train.py > run.log 2>&1
AZURE_CUSTOM_MODEL_ID=... INVOICE_EXPERIMENT=azure_custom_invoice uv run train.py > run.log 2>&1
INVOICE_EXPERIMENT=openrouter_vision uv run train.py > run.log 2>&1
INVOICE_EXPERIMENT=ollama_vision uv run train.py > run.log 2>&1
```

Useful controls:

```bash
INVOICE_DOC_LIMIT=1
MISTRAL_EXTRACT_MODEL=mistral-small-2603
OPENROUTER_MODEL=qwen/qwen2.5-vl-72b-instruct
OLLAMA_MODEL=qwen2.5vl:7b
```

## Candidate Models

Prioritize this exploration order:

1. Mistral OCR plus Mistral Small 4 structured extraction.
2. Mistral OCR table-format variants and prompt/schema tightening.
3. Azure Prebuilt Invoice.
4. OpenRouter VLMs such as Qwen2.5-VL for image invoices.
5. Ollama local VLMs for zero marginal cost image extraction.
6. PaddleOCR v4 as a cheap OCR-first baseline if dependencies are added.
7. DeepSeek-OCR and HunyuanOCR for local/self-hosted OCR comparisons.
8. Donut and LayoutLMv3 if fine-tuning or layout-token preprocessing becomes worthwhile.
9. Azure Custom with `AZURE_CUSTOM_MODEL_ID` once a trained custom model is available.

## Logging

`results.tsv` is tab-separated:

```tsv
commit	accuracy	adjusted_score	cost_per_doc	latency_s	status	description
```

After each run:

```bash
grep "^accuracy:\\|^adjusted_score:\\|^avg_cost_usd:\\|^avg_latency_seconds:\\|^crash_rate:" run.log
```

## Keep Or Revert

Loop:

1. Check branch and current kept commit.
2. Modify only `train.py` for the experiment idea unless intentionally improving fixed setup before the loop begins.
3. Commit the candidate.
4. Run `uv run train.py > run.log 2>&1`.
5. Append a row to `results.tsv`.
6. Keep if `accuracy` improves clearly.
7. Keep if accuracy is tied and `adjusted_score` improves meaningfully.
8. Discard if accuracy drops without a strong adjusted-score reason, or if the run crashes in a way that reflects a bad idea.

The evaluator is frozen once baseline scoring begins.
