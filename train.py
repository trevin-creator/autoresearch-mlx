#!/usr/bin/env python3
"""Invoice OCR extraction experiment runner.

The autoresearch loop mutates this file between experiments. The evaluator in
score_invoices.py is the fixed metric surface.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from score_invoices import score_predictions


DATA_DIR = Path(os.getenv("INVOICE_DATA_DIR", "../Training_invoices"))
PREDICTIONS_PATH = Path(os.getenv("PREDICTIONS_PATH", "predictions.jsonl"))
REPORT_PATH = Path(os.getenv("REPORT_PATH", "invoice_report.json"))
EXPERIMENT = os.getenv("INVOICE_EXPERIMENT", "mistral_ocr_small4_v1")
DOC_LIMIT = int(os.getenv("INVOICE_DOC_LIMIT", "0") or "0")
REQUEST_TIMEOUT = int(os.getenv("INVOICE_REQUEST_TIMEOUT", "180"))

TARGET_FIELDS = [
    "Store",
    "Vendor",
    "Invoice No",
    "Invoice Date",
    "Total Quantity",
    "Bottle Deposit",
    "Invoice Amount",
    "Document Type",
    "Adjustment",
]

TARGET_LINE_FIELDS = [
    "Item Code",
    "Description",
    "Cases",
    "Quantity",
    "Unit Price",
    "Line Amount",
    "Pieces",
    "Discount",
    "Deposit",
    "Deposit Qty",
]

EXTRACTION_PROMPT = f"""Extract invoice data from the document text.

Return only valid JSON with this shape:
{{
  "fields": {{
    "Store": "",
    "Vendor": "",
    "Invoice No": "",
    "Invoice Date": "",
    "Total Quantity": "",
    "Bottle Deposit": "",
    "Invoice Amount": "",
    "Document Type": "",
    "Adjustment": ""
  }},
  "line_items": [
    {{
      "Item Code": "",
      "Description": "",
      "Cases": "",
      "Quantity": "",
      "Unit Price": "",
      "Line Amount": "",
      "Pieces": "",
      "Discount": "",
      "Deposit": "",
      "Deposit Qty": ""
    }}
  ]
}}

Rules:
- Use empty strings for missing fields.
- Preserve exact invoice-visible values when possible.
- Do not invent rows or totals.
- Prefer these header fields: {", ".join(TARGET_FIELDS)}.
- Prefer these line item fields: {", ".join(TARGET_LINE_FIELDS)}.
"""


@dataclass
class ExtractionResult:
    fields: dict[str, Any]
    line_items: list[dict[str, Any]]
    cost_usd: float
    latency_seconds: float
    status: str = "ok"
    error: str = ""


def main() -> None:
    documents = discover_documents(DATA_DIR)
    if DOC_LIMIT > 0:
        documents = documents[:DOC_LIMIT]
    if not documents:
        raise SystemExit(f"No invoice documents found in {DATA_DIR}")

    extractor = build_extractor(EXPERIMENT)
    records = []
    run_started = time.time()

    print(f"experiment: {EXPERIMENT}")
    print(f"data_dir: {DATA_DIR}")
    print(f"documents: {len(documents)}")

    for document_id, path in documents:
        print(f"extracting {document_id}: {path.name}", flush=True)
        started = time.time()
        try:
            result = extractor(path)
        except Exception as exc:  # noqa: BLE001 - experiments should log failures, not hide them.
            result = ExtractionResult(
                fields={},
                line_items=[],
                cost_usd=0.0,
                latency_seconds=time.time() - started,
                status="crash",
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(
            {
                "document_id": document_id,
                "source_path": str(path),
                "fields": result.fields,
                "line_items": result.line_items,
                "cost_usd": result.cost_usd,
                "latency_seconds": result.latency_seconds,
                "status": result.status,
                "error": result.error,
            }
        )

    write_jsonl(PREDICTIONS_PATH, records)
    summary = score_predictions(DATA_DIR, PREDICTIONS_PATH, REPORT_PATH, document_ids=[document_id for document_id, _ in documents])
    total_seconds = time.time() - run_started

    print("---")
    print(f"accuracy:             {summary['accuracy']:.6f}")
    print(f"header_accuracy:      {summary['header_accuracy']:.6f}")
    print(f"line_item_accuracy:   {summary['line_item_accuracy']:.6f}")
    print(f"adjusted_score:       {summary['adjusted_score']:.6f}")
    print(f"docs:                 {summary['docs']}")
    print(f"avg_cost_usd:         {summary['avg_cost_usd']:.6f}")
    print(f"avg_latency_seconds:  {summary['avg_latency_seconds']:.3f}")
    print(f"docs_per_minute:      {summary['docs_per_minute']:.3f}")
    print(f"crash_rate:           {summary['crash_rate']:.6f}")
    print(f"field_matches:        {summary['field_matches']}")
    print(f"field_total:          {summary['field_total']}")
    print(f"total_seconds:        {total_seconds:.1f}")


def discover_documents(data_dir: Path) -> list[tuple[str, Path]]:
    json_ids = sorted(path.stem for path in data_dir.glob("*.json"))
    documents: list[tuple[str, Path]] = []
    for document_id in json_ids:
        candidates = [
            data_dir / f"{document_id}.pdf",
            data_dir / f"{document_id}.png",
            data_dir / f"{document_id}.jpg",
            data_dir / f"{document_id}.jpeg",
        ]
        document_path = next((path for path in candidates if path.exists()), None)
        if document_path is not None:
            documents.append((document_id, document_path))
    return documents


def build_extractor(name: str):
    extractors = {
        "dry_run_empty": dry_run_empty,
        "mistral_ocr_small4_v1": mistral_ocr_small4_v1,
        "mistral_ocr_small4_table_html": mistral_ocr_small4_table_html,
        "mistral_small4_direct_image": mistral_small4_direct_image,
        "azure_prebuilt_invoice": azure_prebuilt_invoice,
        "azure_custom_invoice": azure_custom_invoice,
        "openrouter_vision": openrouter_vision,
        "ollama_vision": ollama_vision,
    }
    if name not in extractors:
        available = ", ".join(sorted(extractors))
        raise SystemExit(f"Unknown INVOICE_EXPERIMENT={name!r}. Available: {available}")
    return extractors[name]


def dry_run_empty(path: Path) -> ExtractionResult:
    started = time.time()
    return ExtractionResult({}, [], 0.0, time.time() - started, status="ok")


def mistral_ocr_small4_v1(path: Path) -> ExtractionResult:
    started = time.time()
    markdown, ocr_cost = run_mistral_ocr(path, table_format=None)
    fields, rows, chat_cost = run_mistral_text_extraction(markdown, model=os.getenv("MISTRAL_EXTRACT_MODEL", "mistral-small-2603"))
    return ExtractionResult(fields, rows, ocr_cost + chat_cost, time.time() - started)


def mistral_ocr_small4_table_html(path: Path) -> ExtractionResult:
    started = time.time()
    markdown, ocr_cost = run_mistral_ocr(path, table_format="html", extract_header=True, extract_footer=True)
    fields, rows, chat_cost = run_mistral_text_extraction(markdown, model=os.getenv("MISTRAL_EXTRACT_MODEL", "mistral-small-2603"))
    return ExtractionResult(fields, rows, ocr_cost + chat_cost, time.time() - started)


def mistral_small4_direct_image(path: Path) -> ExtractionResult:
    if path.suffix.lower() == ".pdf":
        return mistral_ocr_small4_v1(path)
    started = time.time()
    fields, rows, cost = run_mistral_image_extraction(path, model=os.getenv("MISTRAL_EXTRACT_MODEL", "mistral-small-2603"))
    return ExtractionResult(fields, rows, cost, time.time() - started)


def azure_prebuilt_invoice(path: Path) -> ExtractionResult:
    started = time.time()
    payload = azure_analyze_document(path, "prebuilt-invoice")
    fields, rows = parse_azure_invoice(payload)
    cost = float(os.getenv("AZURE_PREBUILT_COST_PER_DOC", "0.01"))
    return ExtractionResult(fields, rows, cost, time.time() - started)


def azure_custom_invoice(path: Path) -> ExtractionResult:
    started = time.time()
    model_id = require_env("AZURE_CUSTOM_MODEL_ID")
    payload = azure_analyze_document(path, model_id)
    fields, rows = parse_azure_custom_invoice(payload)
    cost = float(os.getenv("AZURE_CUSTOM_COST_PER_DOC", os.getenv("AZURE_PREBUILT_COST_PER_DOC", "0.01")))
    return ExtractionResult(fields, rows, cost, time.time() - started)


def azure_analyze_document(path: Path, model_id: str) -> dict[str, Any]:
    endpoint = require_env("AZURE_FORM_RECOGNIZER_ENDPOINT").rstrip("/")
    key = require_env("AZURE_FORM_RECOGNIZER_KEY")
    api_version = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2023-07-31")
    url = f"{endpoint}/formrecognizer/documentModels/{model_id}:analyze?api-version={api_version}"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": mime_type(path),
    }
    response = requests.post(url, headers=headers, data=path.read_bytes(), timeout=REQUEST_TIMEOUT)
    if response.status_code != 202:
        raise RuntimeError(f"Azure analyze failed: {response.status_code} {response.text[:500]}")
    operation_location = response.headers.get("operation-location")
    if not operation_location:
        raise RuntimeError("Azure response did not include operation-location")
    return poll_azure_operation(operation_location, key)


def openrouter_vision(path: Path) -> ExtractionResult:
    if path.suffix.lower() == ".pdf":
        raise RuntimeError("OpenRouter vision experiment needs image input; convert PDFs or use Mistral/Azure for PDFs.")
    started = time.time()
    api_key = require_env("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    data_url = file_data_url(path)
    payload = {
        "model": model,
        "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0")),
        "messages": [
            {"role": "system", "content": "You extract invoice data and return only valid JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost/autoresearch-mlx",
            "X-Title": "invoice-ocr-autoresearch",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter failed: {response.status_code} {response.text[:500]}")
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    fields, rows = parse_extraction_json(content)
    cost = estimate_openrouter_cost(body)
    return ExtractionResult(fields, rows, cost, time.time() - started)


def ollama_vision(path: Path) -> ExtractionResult:
    if path.suffix.lower() == ".pdf":
        raise RuntimeError("Ollama vision experiment needs image input; convert PDFs or use an OCR-first experiment.")
    started = time.time()
    model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
    response = requests.post(
        os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"),
        json={
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT,
                    "images": [base64.b64encode(path.read_bytes()).decode("ascii")],
                }
            ],
            "options": {"temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0"))},
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Ollama failed: {response.status_code} {response.text[:500]}")
    content = response.json()["message"]["content"]
    fields, rows = parse_extraction_json(content)
    return ExtractionResult(fields, rows, 0.0, time.time() - started)


def run_mistral_ocr(path: Path, table_format: str | None, extract_header: bool = False, extract_footer: bool = False) -> tuple[str, float]:
    api_key = require_env("MISTRAL_API_KEY")
    body: dict[str, Any] = {
        "model": os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
        "document": mistral_document_chunk(path),
        "include_image_base64": False,
    }
    if table_format is not None:
        body["table_format"] = table_format
    if extract_header:
        body["extract_header"] = True
    if extract_footer:
        body["extract_footer"] = True

    response = requests.post(
        "https://api.mistral.ai/v1/ocr",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Mistral OCR failed: {response.status_code} {response.text[:500]}")
    payload = response.json()
    pages = payload.get("pages", [])
    markdown = "\n\n".join(page.get("markdown", "") for page in pages if isinstance(page, dict))
    page_count = max(1, len(pages))
    cost = page_count * float(os.getenv("MISTRAL_OCR_COST_PER_PAGE", "0.001"))
    return markdown, cost


def run_mistral_text_extraction(markdown: str, model: str) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    api_key = require_env("MISTRAL_API_KEY")
    payload = {
        "model": model,
        "temperature": float(os.getenv("MISTRAL_TEMPERATURE", "0")),
        "messages": [
            {"role": "system", "content": "You extract invoice data and return only valid JSON."},
            {"role": "user", "content": f"{EXTRACTION_PROMPT}\n\nOCR text:\n{markdown}"},
        ],
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Mistral extraction failed: {response.status_code} {response.text[:500]}")
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    fields, rows = parse_extraction_json(content)
    return fields, rows, estimate_mistral_chat_cost(body)


def run_mistral_image_extraction(path: Path, model: str) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    api_key = require_env("MISTRAL_API_KEY")
    payload = {
        "model": model,
        "temperature": float(os.getenv("MISTRAL_TEMPERATURE", "0")),
        "messages": [
            {"role": "system", "content": "You extract invoice data and return only valid JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": file_data_url(path)},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Mistral image extraction failed: {response.status_code} {response.text[:500]}")
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    fields, rows = parse_extraction_json(content)
    return fields, rows, estimate_mistral_chat_cost(body)


def mistral_document_chunk(path: Path) -> dict[str, str]:
    data_url = file_data_url(path)
    if path.suffix.lower() == ".pdf":
        return {"type": "document_url", "document_url": data_url}
    return {"type": "image_url", "image_url": data_url}


def poll_azure_operation(operation_location: str, key: str) -> dict[str, Any]:
    deadline = time.time() + REQUEST_TIMEOUT
    while time.time() < deadline:
        response = requests.get(operation_location, headers={"Ocp-Apim-Subscription-Key": key}, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(f"Azure poll failed: {response.status_code} {response.text[:500]}")
        payload = response.json()
        status = str(payload.get("status", "")).lower()
        if status == "succeeded":
            return payload
        if status == "failed":
            raise RuntimeError(f"Azure analysis failed: {json.dumps(payload)[:500]}")
        time.sleep(2)
    raise TimeoutError("Timed out waiting for Azure invoice analysis")


def parse_azure_invoice(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    documents = payload.get("analyzeResult", {}).get("documents", [])
    if not documents:
        return {}, []
    fields = documents[0].get("fields", {})

    def value(name: str) -> Any:
        node = fields.get(name, {})
        return node.get("valueString") or node.get("valueNumber") or node.get("valueDate") or node.get("content") or ""

    extracted = {
        "Vendor": value("VendorName"),
        "Invoice No": value("InvoiceId"),
        "Invoice Date": value("InvoiceDate"),
        "Invoice Amount": value("InvoiceTotal") or value("AmountDue"),
        "Total Quantity": value("TotalQuantity"),
    }
    rows = []
    for item in fields.get("Items", {}).get("valueArray", []) or []:
        item_fields = item.get("valueObject", {})
        rows.append(
            {
                "Item Code": azure_field_value(item_fields.get("ProductCode", {})),
                "Description": azure_field_value(item_fields.get("Description", {})),
                "Quantity": azure_field_value(item_fields.get("Quantity", {})),
                "Unit Price": azure_field_value(item_fields.get("UnitPrice", {})),
                "Line Amount": azure_field_value(item_fields.get("Amount", {})),
            }
        )
    return extracted, rows


def parse_azure_custom_invoice(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    documents = payload.get("analyzeResult", {}).get("documents", [])
    if not documents:
        return {}, []
    raw_fields = documents[0].get("fields", {})
    fields: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, node in raw_fields.items():
        if name.lower() in {"items", "rows", "lineitems", "line_items"}:
            for item in node.get("valueArray", []) or []:
                row = {
                    row_name: azure_field_value(row_node)
                    for row_name, row_node in (item.get("valueObject", {}) or {}).items()
                }
                if row:
                    rows.append(row)
        else:
            fields[name] = azure_field_value(node)
    return fields, rows


def azure_field_value(node: dict[str, Any]) -> Any:
    return (
        node.get("valueString")
        or node.get("valueNumber")
        or node.get("valueCurrency", {}).get("amount")
        or node.get("content")
        or ""
    )


def parse_extraction_json(content: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {content[:200]}")
    payload = json.loads(match.group(0))
    fields = payload.get("fields", {})
    if not isinstance(fields, dict):
        fields = {}
    line_items = payload.get("line_items") or payload.get("rows") or payload.get("Rows") or []
    if not isinstance(line_items, list):
        line_items = []
    return fields, [row for row in line_items if isinstance(row, dict)]


def estimate_mistral_chat_cost(body: dict[str, Any]) -> float:
    usage = body.get("usage", {})
    input_tokens = float(usage.get("prompt_tokens", 0) or 0)
    output_tokens = float(usage.get("completion_tokens", 0) or 0)
    input_per_million = float(os.getenv("MISTRAL_INPUT_USD_PER_MILLION", "0.15"))
    output_per_million = float(os.getenv("MISTRAL_OUTPUT_USD_PER_MILLION", "0.60"))
    return input_tokens / 1_000_000 * input_per_million + output_tokens / 1_000_000 * output_per_million


def estimate_openrouter_cost(body: dict[str, Any]) -> float:
    usage = body.get("usage", {})
    input_tokens = float(usage.get("prompt_tokens", 0) or 0)
    output_tokens = float(usage.get("completion_tokens", 0) or 0)
    input_per_1k = float(os.getenv("OPENROUTER_INPUT_USD_PER_1K", "0"))
    output_per_1k = float(os.getenv("OPENROUTER_OUTPUT_USD_PER_1K", "0"))
    return input_tokens / 1000 * input_per_1k + output_tokens / 1000 * output_per_1k


def file_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type(path)};base64,{encoded}"


def mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".tif" or suffix == ".tiff":
        return "image/tiff"
    return "application/octet-stream"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")


if __name__ == "__main__":
    main()
