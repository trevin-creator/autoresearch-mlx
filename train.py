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
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from score_invoices import score_predictions


DATA_DIR = Path(os.getenv("INVOICE_DATA_DIR", "../Training_invoices"))
PREDICTIONS_PATH = Path(os.getenv("PREDICTIONS_PATH", "predictions.jsonl"))
REPORT_PATH = Path(os.getenv("REPORT_PATH", "invoice_report.json"))
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", "results.tsv"))
EXPERIMENT = os.getenv("INVOICE_EXPERIMENT", "mistral_ocr_small4_v1")
DOC_LIMIT = int(os.getenv("INVOICE_DOC_LIMIT", "0") or "0")
REQUEST_TIMEOUT = int(os.getenv("INVOICE_REQUEST_TIMEOUT", "180"))
AUTO_LOG_RESULTS = os.getenv("AUTO_LOG_RESULTS", "1").lower() not in {"0", "false", "no"}

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
    if AUTO_LOG_RESULTS:
        append_results_row(summary)


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
        "paddleocr_v4_regex": paddleocr_v4_regex,
        "paddleocr_v5_regex": paddleocr_v5_regex,
        "paddleocr_v4_mistral": paddleocr_v4_mistral,
        "paddleocr_v5_mistral": paddleocr_v5_mistral,
        "donut_cord_regex": donut_cord_regex,
        "layoutlmv3_invoice_token": layoutlmv3_invoice_token,
        "hunyuanocr_direct": hunyuanocr_direct,
        "deepseek_ocr_regex": deepseek_ocr_regex,
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


def paddleocr_v4_regex(path: Path) -> ExtractionResult:
    return paddleocr_regex(path, "PP-OCRv4")


def paddleocr_v5_regex(path: Path) -> ExtractionResult:
    return paddleocr_regex(path, "PP-OCRv5")


def paddleocr_v4_mistral(path: Path) -> ExtractionResult:
    return paddleocr_mistral(path, "PP-OCRv4")


def paddleocr_v5_mistral(path: Path) -> ExtractionResult:
    return paddleocr_mistral(path, "PP-OCRv5")


def paddleocr_regex(path: Path, ocr_version: str) -> ExtractionResult:
    started = time.time()
    text = run_paddleocr(path, ocr_version)
    fields, rows = parse_invoice_text_heuristic(text)
    return ExtractionResult(fields, rows, 0.0, time.time() - started)


def paddleocr_mistral(path: Path, ocr_version: str) -> ExtractionResult:
    started = time.time()
    text = run_paddleocr(path, ocr_version)
    fields, rows, chat_cost = run_mistral_text_extraction(text, model=os.getenv("MISTRAL_EXTRACT_MODEL", "mistral-small-2603"))
    return ExtractionResult(fields, rows, chat_cost, time.time() - started)


def donut_cord_regex(path: Path) -> ExtractionResult:
    started = time.time()
    text = run_donut(path)
    fields, rows = parse_invoice_text_heuristic(text)
    return ExtractionResult(fields, rows, 0.0, time.time() - started)


def hunyuanocr_direct(path: Path) -> ExtractionResult:
    started = time.time()
    content = run_hunyuanocr(path)
    try:
        fields, rows = parse_extraction_json(content)
    except Exception:
        fields, rows = parse_invoice_text_heuristic(content)
    return ExtractionResult(fields, rows, 0.0, time.time() - started)


def deepseek_ocr_regex(path: Path) -> ExtractionResult:
    started = time.time()
    text = run_deepseek_ocr(path)
    fields, rows = parse_invoice_text_heuristic(text)
    return ExtractionResult(fields, rows, 0.0, time.time() - started)


def layoutlmv3_invoice_token(path: Path) -> ExtractionResult:
    started = time.time()
    fields = run_layoutlmv3_invoice_token(path)
    return ExtractionResult(fields, [], 0.0, time.time() - started)


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


def run_paddleocr(path: Path, ocr_version: str) -> str:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("Install local OCR deps with: uv sync --extra local-ocr") from exc

    images = render_document_images(path)
    kwargs = {
        "lang": os.getenv("PADDLE_LANG", "en"),
        "ocr_version": ocr_version,
        "show_log": False,
    }
    try:
        ocr = PaddleOCR(use_angle_cls=True, **kwargs)
    except TypeError:
        kwargs.pop("show_log", None)
        ocr = PaddleOCR(use_textline_orientation=True, **kwargs)

    chunks: list[str] = []
    for image_path in images:
        if hasattr(ocr, "ocr"):
            result = ocr.ocr(str(image_path), cls=True)
        else:
            result = ocr.predict(str(image_path))
        chunks.extend(extract_text_fragments(result))
    return "\n".join(chunks)


def run_donut(path: Path) -> str:
    try:
        from PIL import Image
        from transformers import DonutProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise RuntimeError("Install HF document deps with: uv sync --extra hf-doc") from exc

    import torch

    image_path = render_document_images(path)[0]
    image = Image.open(image_path).convert("RGB")
    model_id = os.getenv("DONUT_MODEL", "naver-clova-ix/donut-base-finetuned-cord-v2")
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    device = hf_device(torch)
    model.to(device)

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    task_prompt = os.getenv("DONUT_TASK_PROMPT", "<s_cord-v2>")
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=int(os.getenv("DONUT_MAX_LENGTH", "768")),
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=int(os.getenv("DONUT_NUM_BEAMS", "1")),
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    return re.sub(r"<[^>]+>", " ", sequence)


def run_hunyuanocr(path: Path) -> str:
    try:
        from PIL import Image
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install HF document deps with: uv sync --extra hf-doc") from exc

    import torch
    import transformers

    model_id = os.getenv("HUNYUAN_MODEL", "tencent/HunyuanOCR")
    image_path = render_document_images(path)[0]
    image = Image.open(image_path).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model_cls = getattr(transformers, "HunYuanVLForConditionalGeneration", None)
    if model_cls is None:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    device = hf_device(torch)
    dtype = torch.bfloat16 if device in {"cuda", "mps"} else torch.float32
    model = model_cls.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=os.getenv("HUNYUAN_ATTN_IMPLEMENTATION", "eager"),
    )
    model.to(device=device, dtype=dtype)
    model.eval()

    prompt = os.getenv("HUNYUAN_PROMPT", EXTRACTION_PROMPT)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = batch_to_device_dtype(inputs, device, dtype)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=int(os.getenv("HUNYUAN_MAX_NEW_TOKENS", "1024")), do_sample=False)
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        input_ids = inputs.get("inputs")
    if input_ids is not None:
        output_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)]
    return clean_repeated_substrings(processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


def run_deepseek_ocr(path: Path) -> str:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install HF document deps with: uv sync --extra hf-doc") from exc

    import torch

    model_id = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
    image_path = render_document_images(path)[0]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=hf_dtype(torch))
    model = model.eval().to(hf_device(torch))
    prompt = os.getenv("DEEPSEEK_OCR_PROMPT", "<image>\nExtract all text from this invoice.")
    if hasattr(model, "infer"):
        return str(
            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=tempfile.mkdtemp(prefix="deepseek_ocr_"),
                base_size=int(os.getenv("DEEPSEEK_BASE_SIZE", "1024")),
                image_size=int(os.getenv("DEEPSEEK_IMAGE_SIZE", "640")),
                crop_mode=os.getenv("DEEPSEEK_CROP_MODE", "1") not in {"0", "false", "False"},
                save_results=False,
            )
        )
    raise RuntimeError("Loaded DeepSeek OCR model does not expose an infer() method in this Transformers version.")


def run_layoutlmv3_invoice_token(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        from transformers import AutoModelForTokenClassification, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install HF document deps with: uv sync --extra hf-doc") from exc

    import torch

    word_image_path, words, boxes = run_paddleocr_words_boxes(path, os.getenv("LAYOUTLM_OCR_VERSION", "PP-OCRv4"))
    if not words:
        raise RuntimeError("PaddleOCR produced no words for LayoutLMv3.")
    image_path = word_image_path
    image = Image.open(image_path).convert("RGB")
    model_id = os.getenv("LAYOUTLMV3_MODEL", "ngvozdenovic/invoice_extraction")
    processor = AutoProcessor.from_pretrained(model_id, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    device = hf_device(torch)
    model.to(device)

    encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True)
    word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else []
    encoding = {key: value.to(device) for key, value in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1)[0].detach().cpu().tolist()

    fields: dict[str, list[str]] = {}
    previous_word_id = None
    for token_index, label_id in enumerate(predictions):
        if token_index >= len(word_ids):
            continue
        word_id = word_ids[token_index]
        if word_id is None or word_id == previous_word_id or word_id >= len(words):
            continue
        previous_word_id = word_id
        label = model.config.id2label.get(label_id, "O")
        if label == "O":
            continue
        label = re.sub(r"^[BI]-", "", label)
        fields.setdefault(label, []).append(words[word_id])
    return {key: " ".join(value) for key, value in fields.items()}


def render_document_images(path: Path) -> list[Path]:
    if path.suffix.lower() != ".pdf":
        return [path]
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Install PDF rendering deps with: uv sync --extra local-ocr or uv sync --extra hf-doc") from exc

    output_dir = Path(tempfile.mkdtemp(prefix="invoice_pages_"))
    doc = fitz.open(path)
    page_limit = int(os.getenv("PDF_PAGE_LIMIT", "1"))
    dpi = int(os.getenv("PDF_RENDER_DPI", "200"))
    image_paths: list[Path] = []
    for index, page in enumerate(doc):
        if index >= page_limit:
            break
        pix = page.get_pixmap(dpi=dpi)
        image_path = output_dir / f"{path.stem}-{index + 1}.png"
        pix.save(image_path)
        image_paths.append(image_path)
    doc.close()
    return image_paths


def run_paddleocr_words_boxes(path: Path, ocr_version: str) -> tuple[Path, list[str], list[list[int]]]:
    try:
        from PIL import Image
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("Install local OCR deps with: uv sync --extra local-ocr") from exc

    image_path = render_document_images(path)[0]
    image = Image.open(image_path)
    width, height = image.size
    kwargs = {
        "lang": os.getenv("PADDLE_LANG", "en"),
        "ocr_version": ocr_version,
        "show_log": False,
    }
    try:
        ocr = PaddleOCR(use_angle_cls=True, **kwargs)
    except TypeError:
        kwargs.pop("show_log", None)
        ocr = PaddleOCR(use_textline_orientation=True, **kwargs)

    result = ocr.ocr(str(image_path), cls=True) if hasattr(ocr, "ocr") else ocr.predict(str(image_path))
    words: list[str] = []
    boxes: list[list[int]] = []
    for box, text in extract_paddle_box_text(result):
        if not text:
            continue
        words.append(text)
        boxes.append(normalize_layout_box(box, width, height))
    return image_path, words, boxes


def extract_paddle_box_text(value: Any) -> list[tuple[list[list[float]], str]]:
    entries: list[tuple[list[list[float]], str]] = []
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and looks_like_box(value[0]):
            text = ""
            if isinstance(value[1], (list, tuple)) and value[1] and isinstance(value[1][0], str):
                text = value[1][0].strip()
            elif isinstance(value[1], str):
                text = value[1].strip()
            if text:
                entries.append((value[0], text))
            return entries
        for item in value:
            entries.extend(extract_paddle_box_text(item))
    return entries


def looks_like_box(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 4
        and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in value[:4])
    )


def normalize_layout_box(box: list[list[float]], width: int, height: int) -> list[int]:
    xs = [float(point[0]) for point in box[:4]]
    ys = [float(point[1]) for point in box[:4]]
    left = int(max(0, min(1000, round(min(xs) / width * 1000))))
    top = int(max(0, min(1000, round(min(ys) / height * 1000))))
    right = int(max(0, min(1000, round(max(xs) / width * 1000))))
    bottom = int(max(0, min(1000, round(max(ys) / height * 1000))))
    return [left, top, max(left + 1, right), max(top + 1, bottom)]


def extract_text_fragments(value: Any) -> list[str]:
    fragments: list[str] = []
    if value is None:
        return fragments
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, dict):
        for key in ("text", "rec_text", "transcription"):
            if key in value:
                fragments.extend(extract_text_fragments(value[key]))
        for key in ("res", "data", "items", "pages"):
            if key in value:
                fragments.extend(extract_text_fragments(value[key]))
        return fragments
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and isinstance(value[1], tuple) and value[1] and isinstance(value[1][0], str):
            return [value[1][0].strip()]
        for item in value:
            fragments.extend(extract_text_fragments(item))
    return fragments


def parse_invoice_text_heuristic(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat = collapse_text(text)
    fields: dict[str, Any] = {}
    patterns = {
        "Invoice No": r"(?:invoice\s*(?:no|number|#|id)[:\s]*)([A-Z0-9\-]+)",
        "Invoice Date": r"(?:invoice\s*date|date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        "Invoice Amount": r"(?:invoice\s*(?:amount|total)|amount\s*due|total)[:\s$]*([\d,]+\.\d{2})",
        "Total Quantity": r"(?:total\s*(?:quantity|qty))[:\s]*(\d+(?:\.\d+)?)",
        "Bottle Deposit": r"(?:bottle\s*deposit|deposit)[:\s$]*([\d,]+\.\d{2})",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, flat, flags=re.IGNORECASE)
        if match:
            fields[field] = match.group(1)

    vendor_match = re.search(r"(?:vendor|supplier)[:\s]+([A-Z][A-Z0-9 &'.,-]{2,80}?)(?:\s{2,}| invoice| date|$)", flat, flags=re.IGNORECASE)
    if vendor_match:
        fields["Vendor"] = vendor_match.group(1).strip()

    rows: list[dict[str, Any]] = []
    money = r"\d+(?:,\d{3})*(?:\.\d{1,2})?"
    for line in text.splitlines():
        cleaned = collapse_text(line)
        match = re.match(rf"^(\d{{3,}})\s+(.+?)\s+(\d+)\s+(\d+)\s+({money})\s+({money})(?:\s|$)", cleaned)
        if match:
            rows.append(
                {
                    "Item Code": match.group(1),
                    "Description": match.group(2),
                    "Cases": match.group(3),
                    "Quantity": match.group(4),
                    "Unit Price": match.group(5),
                    "Line Amount": match.group(6),
                }
            )
    return fields, rows


def hf_device(torch_module):
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def hf_dtype(torch_module):
    if torch_module.cuda.is_available():
        return torch_module.float16
    return torch_module.float32


def batch_to_device_dtype(batch: Any, device: str, dtype: Any) -> Any:
    if hasattr(batch, "items"):
        return {key: batch_to_device_dtype(value, device, dtype) for key, value in batch.items()}
    if isinstance(batch, list):
        return [batch_to_device_dtype(value, device, dtype) for value in batch]
    if isinstance(batch, tuple):
        return tuple(batch_to_device_dtype(value, device, dtype) for value in batch)
    if hasattr(batch, "to"):
        if getattr(batch, "is_floating_point", lambda: False)():
            return batch.to(device=device, dtype=dtype)
        return batch.to(device=device)
    return batch


def collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_repeated_substrings(text: str) -> str:
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        index = n - length
        while index >= 0 and text[index : index + length] == candidate:
            count += 1
            index -= length
        if count >= 10:
            return text[: n - length * (count - 1)]
    return text


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


def append_results_row(summary: dict[str, Any]) -> None:
    status = os.getenv("RUN_STATUS")
    if not status:
        status = "crash" if summary["crash_rate"] >= 1.0 else "run"
    description = os.getenv("RUN_DESCRIPTION") or default_run_description()
    row = [
        current_commit(),
        f"{summary['accuracy']:.6f}",
        f"{summary['adjusted_score']:.6f}",
        f"{summary['avg_cost_usd']:.6f}",
        f"{summary['avg_latency_seconds']:.3f}",
        sanitize_tsv(status),
        sanitize_tsv(description),
    ]
    if not RESULTS_PATH.exists() or RESULTS_PATH.read_text().strip() == "":
        RESULTS_PATH.write_text("commit\taccuracy\tadjusted_score\tcost_per_doc\tlatency_s\tstatus\tdescription\n")
    with RESULTS_PATH.open("a") as handle:
        handle.write("\t".join(row) + "\n")


def default_run_description() -> str:
    bits = [EXPERIMENT, f"docs={DOC_LIMIT or 'all'}"]
    for name in ("MISTRAL_EXTRACT_MODEL", "OPENROUTER_MODEL", "OLLAMA_MODEL", "PADDLE_LANG"):
        value = os.getenv(name)
        if value:
            bits.append(f"{name.lower()}={value}")
    bits.append(datetime.now(timezone.utc).strftime("utc=%Y-%m-%dT%H:%M:%SZ"))
    return " ".join(bits)


def current_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode != 0
        return f"{commit}-dirty" if dirty or staged else commit
    except Exception:
        return "unknown"


def sanitize_tsv(value: str) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").strip()


if __name__ == "__main__":
    main()
