"""Eight PDF-recommended fields; official template acceptance unverified."""
from __future__ import annotations
import csv

CSV_FIELDS = ["path_to_study","study_uid","image_uid","anatomical_region",
              "quality_class","violation_type","processing_status",
              "time_of_processing"]
REGIONS = {"Поясничный отдел позвоночника", "Проксимальный отдел бедра"}

def validate_csv(path) -> list[str]:
    errors=[]
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != CSV_FIELDS:
            errors.append(f"columns: expected {CSV_FIELDS!r}, got {reader.fieldnames!r}")
        rows=list(reader)
    if not rows: errors.append("empty CSV has no inference results")
    for n,row in enumerate(rows,2):
        if row.get("processing_status") not in {"Success", "Failure"}:
            errors.append(f"row {n}: invalid processing_status")
        if row.get("processing_status") == "Success" and row.get("anatomical_region") not in REGIONS:
            errors.append(f"row {n}: invalid anatomical_region")
        if row.get("processing_status") == "Success" and row.get("quality_class") not in {"0", "1"}:
            errors.append(f"row {n}: invalid quality_class")
        if row.get("processing_status") == "Failure" and any(row.get(k) for k in ("anatomical_region","quality_class","violation_type")):
            errors.append(f"row {n}: failed route must not invent labels")
    return errors
