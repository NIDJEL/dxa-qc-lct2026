"""Local web API for the frozen DXA QC inference service.

The web layer is intentionally thin: all model decisions still come from
``dxa_qc.service.process_study`` and the frozen runtime.  This module only
handles local uploads, job lifecycle, rendering and export.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import Request

from .contract import CSV_FIELDS
from .service import _hash_pixels, process_study, write_official_csv


MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_FILES = 512
JOB_TTL_SECONDS = 6 * 60 * 60


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _safe_relative_name(name: str, index: int) -> Path:
    """Return a safe relative upload path, never allowing traversal."""
    raw = str(name or "").replace("\\", "/")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        parts = [f"upload_{index:04d}.dcm"]
    parts = ["".join(c if c.isalnum() or c in "._- ()" else "_" for c in part)[:128]
             for part in parts]
    clean = Path(*parts)
    if clean.is_absolute() or len(clean.parts) > 24:
        clean = Path(f"upload_{index:04d}.dcm")
    # Uploaded objects are scanned by the existing service's *.dcm discovery.
    if clean.suffix.lower() != ".dcm":
        clean = clean.with_name(clean.name + ".dcm")
    return clean


def _head_names() -> list[str]:
    from . import TARGETS
    return list(TARGETS)


def _result_with_head_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Add named raw-head probabilities without changing existing service keys."""
    names = _head_names()
    thresholds = result.get("provenance", {}).get("thresholds", [])
    for row in result.get("rows", []):
        values = row.get("study_proxy_probabilities") or []
        row["head_probabilities"] = {
            name: float(values[i]) for i, name in enumerate(names) if i < len(values)
        }
        row["head_thresholds"] = {
            name: float(thresholds[i]) for i, name in enumerate(names)
            if i < len(thresholds)
        }
        row["quality_source"] = "independent_E007_quality_head"
        row["result_scope"] = "study_proxy_repeated_per_image"
    return result


def _dicom_metadata(path: Path) -> dict[str, Any]:
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    return {
        "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "") or ""),
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
    }


def build_auto_verified_study_map(root: Path) -> tuple[Path | None, dict[str, Any]]:
    """Build an exact map only when DICOM identifiers establish boundaries.

    The core service remains fail-closed for arbitrary mixed roots.  The web
    layer may provide that service with a generated exact map when every
    decodable file has a non-empty StudyInstanceUID and SeriesInstanceUID,
    series do not cross study boundaries, and each file is covered exactly
    once.  This is not a filename or anatomy heuristic.
    """
    import pydicom

    files = sorted(root.rglob("*.dcm"))
    by_study: dict[str, list[str]] = {}
    study_series: dict[str, set[str]] = {}
    all_series: dict[str, str] = {}
    failures: list[str] = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
            study = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
            series = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
            if not study or not series:
                failures.append(str(path.relative_to(root)))
                continue
            relative = str(path.relative_to(root)).replace("\\", "/")
            by_study.setdefault(study, []).append(relative)
            study_series.setdefault(study, set()).add(series)
            prior = all_series.setdefault(series, study)
            if prior != study:
                failures.append(f"series-crosses-study:{series}")
        except Exception:
            failures.append(str(path.relative_to(root)))
    if len(by_study) <= 1 or failures or set().union(*study_series.values()) != set(all_series):
        return None, {
            "mode": "dicom_uid_only",
            "applied": False,
            "reason": "exact multi-study boundaries were not independently established",
            "failed_files": failures,
        }
    spec = {
        "schema_version": "dxa_study_groups_v1",
        "studies": [
            {"source_study_id": uid, "files": sorted(names)}
            for uid, names in sorted(by_study.items())
        ],
    }
    path = root / ".web_verified_study_groups.json"
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, {
        "mode": "auto_verified_dicom_identifiers",
        "applied": True,
        "study_count": len(by_study),
        "file_count": len(files),
        "reason": "non-empty StudyInstanceUID and SeriesInstanceUID with disjoint series",
    }


def _render_dicom(path: Path, max_side: int = 1600) -> bytes:
    import pydicom
    from PIL import Image, ImageOps

    ds = pydicom.dcmread(str(path), force=False)
    pixels = np.asarray(ds.pixel_array)
    if pixels.ndim != 2:
        raise ValueError("Only single-frame 2-D DICOM images can be previewed")
    values = pixels.astype(np.float32)
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        image = np.zeros(values.shape, dtype=np.uint8)
    else:
        image = np.clip((values - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        image = 255 - image
    pil = Image.fromarray(image, mode="L")
    if max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        pil = pil.resize((max(1, round(pil.width * scale)), max(1, round(pil.height * scale))),
                         Image.Resampling.BICUBIC)
    out = io.BytesIO()
    pil.save(out, format="PNG", optimize=True)
    return out.getvalue()


@dataclass
class Job:
    id: str
    root: Path
    files: dict[str, Path]
    original_names: dict[str, str]
    singleton_mode: bool = False
    status: str = "queued"
    progress: int = 0
    stage: str = "Ожидает запуска"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    grouping: dict[str, Any] = field(default_factory=dict)


class JobManager:
    def __init__(
        self,
        root: Path,
        runtime_factory: Callable[[], Any] | None = None,
        process_fn: Callable[..., dict[str, Any]] = process_study,
        asset_root: Path | None = None,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
        max_files: int = MAX_FILES,
    ):
        self.root = Path(root)
        self.asset_root = Path(asset_root or root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime_factory = runtime_factory
        self.process_fn = process_fn
        self.max_upload_bytes = max_upload_bytes
        self.max_files = max_files
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.runtime_lock = threading.Lock()
        self.runtime: Any | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dxa-qc-web")

    def _get_runtime(self):
        with self.runtime_lock:
            if self.runtime is None:
                self.runtime = self.runtime_factory() if self.runtime_factory else _default_runtime()
            return self.runtime

    def create_job(self, uploads: list[tuple[str, bytes]], singleton_mode: bool = False) -> Job:
        self.cleanup()
        if not uploads:
            raise ValueError("Не передан ни один файл")
        if len(uploads) > self.max_files:
            raise ValueError(f"Слишком много файлов: максимум {self.max_files}")
        total = sum(len(data) for _, data in uploads)
        if total > self.max_upload_bytes:
            raise ValueError(f"Размер загрузки превышает {self.max_upload_bytes // (1024 * 1024)} МБ")
        if any(len(data) > 128 * 1024 * 1024 for _, data in uploads):
            raise ValueError("Размер одного DICOM превышает 128 МБ")
        with self.lock:
            if sum(j.status in {"queued", "running"} for j in self.jobs.values()) >= 3:
                raise ValueError("Очередь заполнена (максимум 3 задания); повторите позже")
        job_id = uuid.uuid4().hex
        job_root = self.root / job_id / "input"
        job_root.mkdir(parents=True, exist_ok=False)
        files: dict[str, Path] = {}
        names: dict[str, str] = {}
        try:
            for index, (name, data) in enumerate(uploads):
                rel = _safe_relative_name(name, index)
                candidate = job_root / rel
                candidate.parent.mkdir(parents=True, exist_ok=True)
                if candidate.exists():
                    candidate = candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}")
                candidate.write_bytes(data)
                image_id = f"image-{index:04d}-{hashlib.sha1(str(rel).encode()).hexdigest()[:8]}"
                files[image_id] = candidate
                names[image_id] = str(candidate.relative_to(job_root)).replace("\\", "/")
            job = Job(job_id, job_root, files, names, singleton_mode=singleton_mode)
            with self.lock:
                self.jobs[job_id] = job
            self.executor.submit(self._run, job_id)
            return job
        except Exception:
            shutil.rmtree(job_root.parent, ignore_errors=True)
            raise

    def _run(self, job_id: str):
        with self.lock:
            job = self.jobs[job_id]
            job.status, job.progress, job.stage, job.started_at = "running", 5, "Проверка DICOM и границ исследований", time.time()
        try:
            map_path, grouping = build_auto_verified_study_map(job.root)
            job.grouping = grouping
            if job.singleton_mode:
                spec = {
                    "schema_version": "dxa_study_groups_v1",
                    "studies": [
                        {"source_study_id": f"singleton:{image_id}", "files": [name]}
                        for image_id, name in job.original_names.items()
                    ],
                }
                map_path = job.root / ".singleton_study_groups.json"
                map_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
                job.grouping = {
                    "mode": "experimental_singleton_bag",
                    "applied": True,
                    "reason": "each uploaded image is forced into its own experimental bag",
                    "official_export_eligible": False,
                }
            elif len(job.files) == 1:
                # The single-file input is independently processable even if
                # the vendor omitted study/series UIDs.
                only_id, only_name = next(iter(job.original_names.items()))
                spec = {"schema_version": "dxa_study_groups_v1", "studies": [
                    {"source_study_id": f"single_input:{only_id}", "files": [only_name]}
                ]}
                map_path = job.root / ".single_input_study_groups.json"
                map_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
                job.grouping = {
                    "mode": "single_input_bag",
                    "applied": True,
                    "reason": "one uploaded file; preliminary study-level proxy",
                    "official_export_eligible": False,
                }
            else:
                # Partial grouping isolates ambiguous/bad groups. The generated
                # exact map is retained as local audit evidence, not handed to
                # the fail-whole-batch CLI path.
                map_path = None
            with self.lock:
                job.progress, job.stage = 18, "Загрузка локальных моделей (один раз на процесс)"
            def progress(stage: str, done: int, total: int):
                with self.lock:
                    if stage == "dicom_read":
                        job.progress, job.stage = 27, f"DICOM проверены: {done}/{total}"
                    else:
                        job.progress = 30 + int(66 * done / max(total, 1))
                        job.stage = f"Нейросетевой анализ исследований: {done}/{total}"
            result = self.process_fn(
                job.root,
                runtime=self._get_runtime(),
                root=self.asset_root,
                study_groups_path=map_path,
                web_auto_groups=map_path is None,
                progress_callback=progress,
            )
            result = _result_with_head_metadata(result)
            result["web_job_id"] = job.id
            result["web_grouping"] = job.grouping
            result["web_export"] = {
                "csv_fields": CSV_FIELDS,
                "official_template_confirmed": False,
                "singleton_export_allowed": False,
                "official_export_eligible": job.grouping.get("official_export_eligible", True)
                    and not job.singleton_mode,
                "failure_unknown_policy_confirmed": False,
            }
            for row in result.get("rows", []):
                matched = next((image_id for image_id, path in job.files.items()
                                if str(path) == row.get("image_path")), None)
                if matched:
                    row["web_image_id"] = matched
                    row["relative_path"] = job.original_names[matched]
                    row["internal_image_path"] = row.get("image_path")
                    row["image_path"] = job.original_names[matched]
                    row["path_to_study"] = str(Path(job.original_names[matched]).parent).replace("\\", "/")
                    if row["path_to_study"] == ".":
                        row["path_to_study"] = ""
            with self.lock:
                job.result = result
                job.status = "completed" if result.get("processing_status") in {"ok", "partial_failure"} else "failed"
                job.progress, job.stage, job.finished_at = 100, "Анализ завершён", time.time()
                if result.get("processing_status") == "partial_failure":
                    job.stage = "Анализ завершён с отдельными ошибками"
        except Exception as exc:
            with self.lock:
                job.status, job.progress, job.stage, job.error, job.finished_at = "failed", 100, "Ошибка анализа", str(exc), time.time()

    def get(self, job_id: str) -> Job:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]

    def delete(self, job_id: str) -> None:
        with self.lock:
            job = self.get(job_id)
            if job.status in {"queued", "running"}:
                raise ValueError("Дождитесь завершения задания перед удалением")
            del self.jobs[job_id]
        shutil.rmtree(job.root.parent, ignore_errors=True)

    def cleanup(self):
        now = time.time()
        with self.lock:
            expired = [j for j in self.jobs.values()
                       if j.finished_at and now - j.finished_at > JOB_TTL_SECONDS]
            for job in expired:
                self.jobs.pop(job.id, None)
                shutil.rmtree(job.root.parent, ignore_errors=True)


def _default_runtime():
    from .service import Runtime
    root = Path(os.environ.get("DXA_QC_ROOT", Path(__file__).parents[2]))
    checkpoint = os.environ.get(
        "DXA_QC_CHECKPOINT",
        "artifacts/E007_FULL_V1/A3/seed_17/fold_0/checkpoints/best.pt",
    )
    return Runtime(root, checkpoint, os.environ.get("DXA_QC_DEVICE", "auto"))


def create_app(root: str | Path | None = None, manager: JobManager | None = None):
    from fastapi import FastAPI, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    project_root = Path(root or os.environ.get("DXA_QC_ROOT", Path(__file__).parents[2])).resolve()
    jobs_root = Path(os.environ.get("DXA_QC_JOBS_DIR", project_root / "tmp" / "dxa_qc_web_jobs"))
    manager = manager or JobManager(jobs_root, asset_root=project_root)
    static_dir = Path(__file__).with_name("web_static")
    app = FastAPI(title="DXA QC локальный анализ", version="1.0")
    app.state.manager = manager

    @app.get("/api/health")
    def health():
        manager.cleanup()
        with manager.lock:
            active = sum(job.status in {"queued", "running"} for job in manager.jobs.values())
            loaded = manager.runtime is not None
        return {
            "status": "ok",
            "service": "dxa-qc-web",
            "runtime_loaded": loaded,
            "active_jobs": active,
            "queued_jobs": sum(job.status == "queued" for job in manager.jobs.values()),
            "device": str(getattr(getattr(manager, "runtime", None), "device", os.environ.get("DXA_QC_DEVICE", "auto"))),
            "offline": True,
        }

    @app.post("/api/jobs")
    async def create_job(request: Request):
        if int(request.headers.get("content-length", "0") or "0") > manager.max_upload_bytes + 2 * 1024 * 1024:
            raise HTTPException(413, "Запрос слишком большой")
        form = await request.form(max_files=manager.max_files, max_fields=8)
        files = form.getlist("files")
        singleton_mode = str(form.get("singleton_mode", "false")).lower() in {"1", "true", "yes", "on"}
        uploads = []
        for upload in files:
            if not isinstance(upload, UploadFile) and not hasattr(upload, "read"):
                continue
            data = await upload.read(manager.max_upload_bytes + 1)
            uploads.append((upload.filename or "", data))
        try:
            job = manager.create_job(uploads, singleton_mode=singleton_mode)
        except ValueError as exc:
            raise HTTPException(413, str(exc)) from exc
        return {"job_id": job.id, "status": job.status, "file_count": len(job.files)}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        try:
            manager.delete(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Задание не найдено") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"deleted": True, "job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        try:
            job = manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Задание не найдено") from exc
        with manager.lock:
            result = job.result
            return {
                "job_id": job.id, "status": job.status, "progress": job.progress,
                "stage": job.stage, "file_count": len(job.files),
                "created_at": job.created_at, "started_at": job.started_at,
                "finished_at": job.finished_at, "error": job.error,
                "summary": summarize(result) if result else None,
            }

    @app.get("/api/jobs/{job_id}/results")
    def job_results(job_id: str):
        try:
            job = manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Задание не найдено") from exc
        if job.result is None:
            raise HTTPException(409, "Результаты ещё не готовы")
        result = json.loads(json.dumps(job.result, default=_json_default))
        result["api_links"] = {
            "csv": f"/api/jobs/{job_id}/export.csv",
            "json": f"/api/jobs/{job_id}/export.json",
        }
        return result

    @app.get("/api/jobs/{job_id}/images/{image_id}")
    def image(job_id: str, image_id: str):
        try:
            job = manager.get(job_id)
            path = job.files[image_id]
            return Response(_render_dicom(path), media_type="image/png",
                            headers={"Cache-Control": "private, max-age=3600"})
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, "Изображение не найдено") from exc
        except Exception as exc:
            raise HTTPException(422, f"Не удалось отобразить DICOM: {exc}") from exc

    @app.get("/api/jobs/{job_id}/download/{image_id}")
    def original(job_id: str, image_id: str):
        try:
            job = manager.get(job_id)
            return FileResponse(job.files[image_id], media_type="application/dicom",
                                filename=job.original_names[image_id])
        except KeyError as exc:
            raise HTTPException(404, "Изображение не найдено") from exc

    @app.get("/api/jobs/{job_id}/export.csv")
    def export_csv(job_id: str, allow_experimental: bool = False):
        try:
            job = manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Задание не найдено") from exc
        if not job.result:
            raise HTTPException(409, "Результаты ещё не готовы")
        if (job.singleton_mode or job.grouping.get("official_export_eligible") is False) and not allow_experimental:
            raise HTTPException(409, "Экспорт singleton-bag отключён без явно заданной политики")
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in CSV_FIELDS}
                         for row in job.result.get("rows", [])
                         if row.get("processing_status") == "Success")
        headers = {"Content-Disposition": f'attachment; filename="dxa_qc_{job_id}.csv"'}
        headers["X-DXA-QC-Export-Warning"] = (
            "experimental-singleton-bag" if job.singleton_mode
            or job.grouping.get("official_export_eligible") is False
            else "organizer-template-and-failure-policy-unconfirmed"
        )
        return Response("\ufeff" + out.getvalue(), media_type="text/csv; charset=utf-8", headers=headers)

    @app.get("/api/jobs/{job_id}/export.json")
    def export_json(job_id: str):
        try:
            job = manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Задание не найдено") from exc
        if not job.result:
            raise HTTPException(409, "Результаты ещё не готовы")
        headers = {"Content-Disposition": f'attachment; filename="dxa_qc_{job_id}.json"'}
        return JSONResponse(job.result, headers=headers)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


def summarize(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    rows = result.get("rows", [])
    return {
        "processed_images": len(rows),
        "spine_images": sum("позвоночник" in str(r.get("anatomical_region", "")).lower() for r in rows),
        "hip_images": sum("бедра" in str(r.get("anatomical_region", "")).lower() for r in rows),
        "quality_positive": sum(r.get("quality_class") == 0 for r in rows),
        "requires_attention": sum(
            r.get("processing_status") != "Success"
            or r.get("quality_class") == 1
            or bool(r.get("violation_type"))
            for r in rows
        ),
        "errors": len(result.get("errors", [])),
        "studies": result.get("study_count", 0),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }


app = create_app()
