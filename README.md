# invoice-ocr-autoresearch

Autonomous experiment loop for invoice OCR and structured data extraction.

The goal is to maximize extraction `accuracy` on a labeled invoice set. The runner also reports an `adjusted_score` that penalizes cost, latency, low throughput, and crashes, but accuracy remains the primary metric.

This repo is designed for “bring your own keys”: no credentials, invoices, predictions, logs, or generated reports are committed.

## What Is Included

- `train.py` - mutable experiment runner. Autoresearch edits this file.
- `score_invoices.py` - fixed evaluator. Treat this as read-only during a run.
- `program.md` - protocol for the autonomous keep/discard loop.
- `results.tsv` - run history.
- `pyproject.toml` / `uv.lock` - reproducible Python environment.

## Data Layout

By default, the runner expects invoices one folder above the repo:

```text
../Training_invoices/
  1.pdf
  1.json
  2.pdf
  2.json
```

Each document file should have a matching label JSON with the same stem. The current evaluator supports simple labels with header fields plus `Rows`, for example:

```json
{
  "Vendor": "Acme",
  "Invoice No": "123",
  "Invoice Date": "4/13/2026",
  "Invoice Amount": "748.05",
  "Rows": [
    {"Item Code": "1103", "Description": "Water", "Quantity": "10"}
  ]
}
```

Override the location with:

```bash
export INVOICE_DATA_DIR=/path/to/Training_invoices
```

## Setup

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For local OCR experiments:

```bash
uv sync --extra local-ocr
```

For Hugging Face document models such as Donut, LayoutLMv3, HunyuanOCR, and DeepSeek-OCR:

```bash
uv sync --extra local-ocr --extra hf-doc
```

The `hf-doc` extra pins the source Transformers commit required by HunyuanOCR.

## Credentials

Export only the keys you intend to use. Do not commit keys.

```bash
export MISTRAL_API_KEY=...
export OPENROUTER_API_KEY=...
export AZURE_FORM_RECOGNIZER_ENDPOINT=...
export AZURE_FORM_RECOGNIZER_KEY=...
export HF_TOKEN=...
export AZURE_CUSTOM_MODEL_ID=...
```

On macOS with zsh, put personal exports in your home config, not in this repo:

```bash
/Users/<you>/.zshrc
```

Then reload:

```bash
source ~/.zshrc
```

## Run One Experiment

Start with one invoice while developing:

```bash
INVOICE_DOC_LIMIT=1 \
INVOICE_EXPERIMENT=mistral_ocr_small4_v1 \
RUN_DESCRIPTION="Mistral OCR plus structured extraction" \
uv run train.py > run.log 2>&1
```

Useful experiments:

```bash
INVOICE_EXPERIMENT=mistral_ocr_small4_v1
INVOICE_EXPERIMENT=mistral_ocr_small4_table_html
INVOICE_EXPERIMENT=paddleocr_v4_mistral
INVOICE_EXPERIMENT=azure_prebuilt_invoice
INVOICE_EXPERIMENT=openrouter_vision
INVOICE_EXPERIMENT=paddleocr_v4_regex
INVOICE_EXPERIMENT=donut_cord_regex
INVOICE_EXPERIMENT=layoutlmv3_invoice_token
INVOICE_EXPERIMENT=hunyuanocr_direct
INVOICE_EXPERIMENT=deepseek_ocr_regex
```

For HunyuanOCR on Apple Silicon, use lower render DPI to avoid MPS memory errors:

```bash
PDF_RENDER_DPI=96 HUNYUAN_MAX_NEW_TOKENS=512
```

## Tracking Runs

`train.py` appends every run to `results.tsv` unless disabled:

```bash
AUTO_LOG_RESULTS=0 uv run train.py
```

After a run:

```bash
tail -n 5 results.tsv
tail -n 80 run.log
```

The TSV columns are:

```tsv
commit	accuracy	adjusted_score	cost_per_doc	latency_s	status	description
```

Use `keep` for improvements, `discard` for losing ideas, and `crash` for failed runs. The loop should prefer higher `accuracy`; use `adjusted_score` as a tie-breaker.

## Current Practical Notes

- Mistral OCR plus structured extraction is the strongest current baseline.
- PaddleOCR v4 works locally; pairing it with Mistral extraction is more useful than regex-only parsing.
- Azure Prebuilt is wired, but requires Azure credentials.
- HunyuanOCR can run locally with source Transformers and low DPI, but current output quality is weak on the sample invoice.
- DeepSeek-OCR currently has Transformers/model-code compatibility issues on this Mac path and is better suited to a separate remote GPU environment.

## License

MIT. See [LICENSE](LICENSE).
