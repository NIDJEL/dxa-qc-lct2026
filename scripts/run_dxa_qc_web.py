"""Run the local DXA QC web service.

This is deliberately separate from ``run_dxa_qc_service.py`` so the
competition eight-field CLI remains unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DXA_QC_ROOT", str(ROOT))

from dxa_qc.web import app  # noqa: E402


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("DXA_QC_HOST", "127.0.0.1"),
        port=int(os.environ.get("DXA_QC_PORT", "8000")),
        log_level=os.environ.get("DXA_QC_LOG_LEVEL", "info"),
    )
