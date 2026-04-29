#!/usr/bin/env python3
"""Fixed invoice extraction scorer for the autoresearch loop."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


EXCLUDED_TOP_LEVEL_FIELDS = {
    "Batch ID",
    "Link",
    "Error Message",
    "Error Fields",
    "File Name",
}

EXCLUDED_LINE_ITEM_FIELDS = {
    "Error Fields",
}

NUMERIC_HINTS = (
    "amount",
    "price",
    "quantity",
    "qty",
    "discount",
    "deposit",
    "tax",
    "total",
    "surcharge",
    "pieces",
    "cases",
    "unitpercase",
    "calcvalue",
    "upc",
)

BOOLEAN_HINTS = (
    "invalid",
    "cog",
    "holiday",
)

DATE_HINTS = ("date",)

HEADER_ALIASES = {
    "store": "store",
    "location": "store",
    "vendor": "vendor",
    "vendorname": "vendor",
    "supplier": "vendor",
    "suppliername": "vendor",
    "invoiceno": "invoice_no",
    "invoicenumber": "invoice_no",
    "invoiceid": "invoice_no",
    "invoice": "invoice_no",
    "invoicedate": "invoice_date",
    "date": "invoice_date",
    "totalquantity": "total_quantity",
    "totalqty": "total_quantity",
    "bottledeposit": "bottle_deposit",
    "depositamount": "bottle_deposit",
    "invoiceamount": "invoice_amount",
    "invoicetotal": "invoice_amount",
    "amountdue": "invoice_amount",
    "total": "invoice_amount",
    "documenttype": "document_type",
    "doctype": "document_type",
    "adjustment": "adjustment",
}

LINE_ALIASES = {
    "itemcode": "item_code",
    "productcode": "item_code",
    "sku": "item_code",
    "item": "item_code",
    "description": "description",
    "itemdescription": "description",
    "productdescription": "description",
    "cases": "cases",
    "case": "cases",
    "quantity": "quantity",
    "qty": "quantity",
    "unitprice": "unit_price",
    "price": "unit_price",
    "lineamount": "line_amount",
    "amount": "line_amount",
    "linetotal": "line_amount",
    "pieces": "pieces",
    "piece": "pieces",
    "discount": "discount",
    "deposit": "deposit",
    "depositqty": "deposit_qty",
    "depositquantity": "deposit_qty",
}


@dataclass
class DocumentMetrics:
    header_matches: int = 0
    header_total: int = 0
    line_matches: int = 0
    line_total: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    crashed: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--document-id", action="append", help="Score only this document id. May be repeated.")
    args = parser.parse_args()

    summary = score_predictions(
        Path(args.ground_truth_dir),
        Path(args.predictions),
        Path(args.report_json) if args.report_json else None,
        document_ids=args.document_id,
    )
    print_summary(summary)


def score_predictions(
    ground_truth_dir: Path,
    predictions_path: Path,
    report_path: Path | None = None,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    gt_paths = sorted(ground_truth_dir.glob("*.json"))
    if document_ids is not None:
        allowed_ids = {str(document_id) for document_id in document_ids}
        gt_paths = [path for path in gt_paths if path.stem in allowed_ids]
    predictions = load_predictions(predictions_path)
    documents: dict[str, DocumentMetrics] = {}

    for gt_path in gt_paths:
        document_id = gt_path.stem
        expected_fields, expected_line_items = parse_ground_truth(gt_path)
        predicted = predictions.get(document_id)
        metrics = DocumentMetrics()
        if predicted is None:
            metrics.crashed = True
            metrics.header_total = len(expected_fields)
            metrics.line_total = sum(len(row) for row in expected_line_items)
            documents[document_id] = metrics
            continue

        header_matches, header_total = compare_fields(expected_fields, predicted["fields"])
        line_matches, line_total = compare_line_items(expected_line_items, predicted["line_items"])
        metrics.header_matches = header_matches
        metrics.header_total = header_total
        metrics.line_matches = line_matches
        metrics.line_total = line_total
        metrics.cost_usd = predicted["cost_usd"]
        metrics.latency_seconds = predicted["latency_seconds"]
        metrics.crashed = bool(predicted["crashed"])
        documents[document_id] = metrics

    summary = summarize_documents(documents)
    if report_path is not None:
        report_path.write_text(json.dumps({"summary": summary, "documents": serialize_documents(documents)}, indent=2, sort_keys=True))
    return summary


def parse_ground_truth(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("Ch"), list):
        return parse_grooper_payload(payload)
    if isinstance(payload, dict):
        return parse_plain_invoice_payload(payload)
    return {}, []


def parse_plain_invoice_payload(payload: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    fields = {key: value for key, value in payload.items() if key not in {"Rows", "rows", "line_items"}}
    rows = payload.get("Rows") or payload.get("rows") or payload.get("line_items") or []
    if not isinstance(rows, list):
        rows = []
    return normalize_fields(fields, EXCLUDED_TOP_LEVEL_FIELDS, HEADER_ALIASES), normalize_line_items(rows)


def parse_grooper_payload(payload: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    fields: dict[str, Any] = {}
    line_items: list[dict[str, Any]] = []
    for child in payload.get("Ch", []):
        child_type = child.get("__type", "")
        name = child.get("Name")
        if child_type.startswith("FieldInstance") and name:
            value = extract_field_value(child)
            if value not in (None, ""):
                fields[name] = value
        elif child_type.startswith("TableInstance") and name == "Line Items":
            for row in child.get("Ch", []):
                if not row.get("__type", "").startswith("TableRowInstance"):
                    continue
                row_fields: dict[str, Any] = {}
                for cell in row.get("Cells", []):
                    cell_name = cell.get("Name")
                    if not cell_name:
                        continue
                    value = cell.get("Val")
                    if value not in (None, ""):
                        row_fields[cell_name] = value
                if row_fields:
                    line_items.append(row_fields)
    return normalize_fields(fields, EXCLUDED_TOP_LEVEL_FIELDS, HEADER_ALIASES), normalize_line_items(line_items)


def extract_field_value(node: dict[str, Any]) -> Any:
    if "Val" in node and node["Val"] not in (None, ""):
        return node["Val"]
    for alt in node.get("AE", []):
        if alt.get("Val") not in (None, ""):
            return alt["Val"]
    return None


def parse_prediction_record(record: dict[str, Any]) -> tuple[str | None, dict[str, str], list[dict[str, str]], float, float, bool]:
    document_id = record.get("document_id") or record.get("id")
    if not document_id:
        source_path = record.get("source_path") or record.get("file_name")
        if source_path:
            document_id = Path(source_path).stem
    if document_id is not None:
        document_id = str(document_id)

    fields = record.get("fields", {})
    if not isinstance(fields, dict):
        fields = {}
    line_items = record.get("line_items") or record.get("Rows") or record.get("rows") or []
    if not isinstance(line_items, list):
        line_items = []

    cost_usd = safe_float(record.get("cost_usd", 0.0))
    latency_seconds = safe_float(record.get("latency_seconds", 0.0))
    status = str(record.get("status", "ok")).lower()
    crashed = status not in {"ok", "success"}

    return (
        document_id,
        normalize_fields(fields, EXCLUDED_TOP_LEVEL_FIELDS, HEADER_ALIASES),
        normalize_line_items(line_items),
        cost_usd,
        latency_seconds,
        crashed,
    )


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return {}
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        records = parsed if isinstance(parsed, list) else [parsed]

    predictions: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        document_id, fields, line_items, cost_usd, latency_seconds, crashed = parse_prediction_record(record)
        if document_id is None:
            continue
        predictions[document_id] = {
            "fields": fields,
            "line_items": line_items,
            "cost_usd": cost_usd,
            "latency_seconds": latency_seconds,
            "crashed": crashed,
        }
    return predictions


def normalize_fields(fields: dict[str, Any], excluded: set[str], aliases: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, raw_value in fields.items():
        if key in excluded:
            continue
        canonical_key = canonical_key_name(key, aliases)
        value = canonical_value(canonical_key, raw_value)
        if value is not None:
            normalized[canonical_key] = value
    return normalized


def normalize_line_items(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = normalize_fields(item, EXCLUDED_LINE_ITEM_FIELDS, LINE_ALIASES)
        if normalized:
            rows.append(normalized)
    return rows


def canonical_key_name(key: str, aliases: dict[str, str]) -> str:
    compact = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return aliases.get(compact, compact)


def canonical_value(field_name: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return canonical_number(str(value))
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True)
    cleaned = collapse_ws(value)
    if cleaned == "":
        return None

    lowered_name = field_name.lower()
    if any(hint in lowered_name for hint in BOOLEAN_HINTS):
        normalized = canonical_bool(cleaned)
        if normalized is not None:
            return normalized
    if any(hint in lowered_name for hint in DATE_HINTS):
        normalized = canonical_date(cleaned)
        if normalized is not None:
            return normalized
    if any(hint in lowered_name for hint in NUMERIC_HINTS):
        normalized = canonical_number(cleaned)
        if normalized is not None:
            return normalized

    normalized = canonical_date(cleaned)
    if normalized is not None and any(char.isdigit() for char in cleaned):
        return normalized
    normalized = canonical_number(cleaned)
    if normalized is not None and re.fullmatch(r"[$,0-9.\-()% ]+", cleaned):
        return normalized
    return collapse_ws(cleaned).upper()


def collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def canonical_bool(value: str) -> str | None:
    lowered = collapse_ws(value).lower()
    if lowered in {"true", "yes", "y", "1"}:
        return "true"
    if lowered in {"false", "no", "n", "0"}:
        return "false"
    return None


def canonical_number(value: str) -> str | None:
    cleaned = collapse_ws(value).replace("$", "").replace(",", "").replace("%", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    if not cleaned:
        return None
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def canonical_date(value: str) -> str | None:
    cleaned = collapse_ws(value)
    if not cleaned:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%m.%d.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def compare_fields(expected: dict[str, str], actual: dict[str, str]) -> tuple[int, int]:
    total = len(expected)
    matches = sum(1 for key, value in expected.items() if actual.get(key) == value)
    return matches, total


def compare_line_items(expected: list[dict[str, str]], actual: list[dict[str, str]]) -> tuple[int, int]:
    matches = 0
    total = sum(len(row) for row in expected)
    unused_actual = set(range(len(actual)))
    for expected_row in expected:
        best_index = None
        best_matches = -1
        for actual_index in unused_actual:
            row_matches = count_field_matches(expected_row, actual[actual_index])
            if row_matches > best_matches:
                best_matches = row_matches
                best_index = actual_index
        if best_index is not None:
            unused_actual.remove(best_index)
            matches += max(0, best_matches)
    return matches, total


def count_field_matches(expected: dict[str, str], actual: dict[str, str]) -> int:
    return sum(1 for key, value in expected.items() if actual.get(key) == value)


def summarize_documents(documents: dict[str, DocumentMetrics]) -> dict[str, Any]:
    total_header_matches = sum(metric.header_matches for metric in documents.values())
    total_header = sum(metric.header_total for metric in documents.values())
    total_line_matches = sum(metric.line_matches for metric in documents.values())
    total_line = sum(metric.line_total for metric in documents.values())
    total_matches = total_header_matches + total_line_matches
    total_fields = total_header + total_line

    accuracy = total_matches / total_fields if total_fields else 0.0
    header_accuracy = total_header_matches / total_header if total_header else 0.0
    line_item_accuracy = total_line_matches / total_line if total_line else 0.0

    doc_count = len(documents)
    total_cost = sum(metric.cost_usd for metric in documents.values())
    total_latency = sum(metric.latency_seconds for metric in documents.values())
    avg_cost_usd = total_cost / doc_count if doc_count else 0.0
    avg_latency_seconds = total_latency / doc_count if doc_count else 0.0
    docs_per_minute = 60.0 / avg_latency_seconds if avg_latency_seconds > 0 else float("inf")
    crash_count = sum(1 for metric in documents.values() if metric.crashed)
    crash_rate = crash_count / doc_count if doc_count else 0.0
    adjusted_score = compute_adjusted_score(accuracy, avg_cost_usd, avg_latency_seconds, docs_per_minute, crash_rate)

    return {
        "accuracy": accuracy,
        "header_accuracy": header_accuracy,
        "line_item_accuracy": line_item_accuracy,
        "adjusted_score": adjusted_score,
        "docs": doc_count,
        "avg_cost_usd": avg_cost_usd,
        "avg_latency_seconds": avg_latency_seconds,
        "docs_per_minute": docs_per_minute,
        "crash_rate": crash_rate,
        "field_matches": total_matches,
        "field_total": total_fields,
    }


def compute_adjusted_score(
    accuracy: float,
    avg_cost_usd: float,
    avg_latency_seconds: float,
    docs_per_minute: float,
    crash_rate: float,
) -> float:
    cost_penalty = min(0.12, avg_cost_usd * 4.0)
    latency_penalty = min(0.08, avg_latency_seconds / 120.0)
    throughput_penalty = min(0.05, max(0.0, 20.0 - docs_per_minute) / 200.0)
    crash_penalty = min(0.20, crash_rate * 0.20)
    return max(0.0, accuracy - cost_penalty - latency_penalty - throughput_penalty - crash_penalty)


def serialize_documents(documents: dict[str, DocumentMetrics]) -> dict[str, Any]:
    return {
        document_id: {
            "header_matches": metric.header_matches,
            "header_total": metric.header_total,
            "line_matches": metric.line_matches,
            "line_total": metric.line_total,
            "cost_usd": metric.cost_usd,
            "latency_seconds": metric.latency_seconds,
            "crashed": metric.crashed,
        }
        for document_id, metric in documents.items()
    }


def print_summary(summary: dict[str, Any]) -> None:
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


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
