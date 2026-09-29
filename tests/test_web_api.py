from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from dxa_qc.service import _hash_pixels, process_study
from dxa_qc.web import JobManager, create_app


def _dicom_bytes(study_uid: str, value: int = 80) -> bytes:
    import tempfile

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    path = Path(tempfile.mktemp(suffix=".dcm"))
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.Rows = ds.Columns = 32
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.full((32, 32), value, dtype=np.uint8).tobytes()
    ds.save_as(path)
    data = path.read_bytes()
    path.unlink()
    return data


class FakeRuntime:
    device = "cpu"
    thresholds = np.full(10, 0.5)
    provenance = {
        "checkpoint": "fake-test-checkpoint",
        "thresholds": [0.5] * 10,
        "temperatures": [1.0] * 10,
        "router_sha256": "test-router",
    }

    def predict_bag_routed(self, pixels):
        return (
            np.array([0.1, 0.2, 0.3, 0.7, 0.2, 0.6, 0.1, 0.2, 0.1, 0.1]),
            0.01,
            {_hash_pixels(pixels[0]): ("hip", "frozen_linear_router", 0.2)},
        )


def _manager(tmp_path):
    return JobManager(
        tmp_path / "jobs",
        asset_root=tmp_path,
        runtime_factory=FakeRuntime,
        process_fn=process_study,
    )


def _wait(client, job_id):
    for _ in range(100):
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in {"completed", "failed"}:
            return data
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_web_upload_real_service_result_and_exports(tmp_path):
    app = create_app(tmp_path, manager=_manager(tmp_path))
    client = TestClient(app)
    payload = _dicom_bytes(generate_uid())
    response = client.post(
        "/api/jobs",
        files={"files": ("test.dcm", payload, "application/dicom")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = _wait(client, job_id)
    assert status["status"] == "completed"
    result = client.get(f"/api/jobs/{job_id}/results").json()
    assert result["row_count"] == 1
    assert result["rows"][0]["anatomical_region"] == "Проксимальный отдел бедра"
    assert result["rows"][0]["quality_source"] == "independent_E007_quality_head"
    assert result["rows"][0]["head_probabilities"]["right_hip_positioning_rotation"] == 0.7
    assert client.get(f"/api/jobs/{job_id}/images/{result['rows'][0]['web_image_id']}").headers["content-type"] == "image/png"
    assert client.get(f"/api/jobs/{job_id}/export.csv").status_code == 409
    assert client.get(f"/api/jobs/{job_id}/export.json").status_code == 200


def test_web_folder_keeps_two_uid_bags_and_partial_failures(tmp_path):
    manager = _manager(tmp_path)
    app = create_app(tmp_path, manager=manager)
    client = TestClient(app)
    files = [
        ("a/one.dcm", _dicom_bytes(generate_uid(), 50)),
        ("b/two.dcm", _dicom_bytes(generate_uid(), 100)),
    ]
    response = client.post(
        "/api/jobs",
        files=[("files", (name, data, "application/dicom")) for name, data in files],
    )
    status = _wait(client, response.json()["job_id"])
    assert status["status"] == "completed"
    result = client.get(f"/api/jobs/{response.json()['job_id']}/results").json()
    assert result["study_count"] == 2
    assert result["grouping_method"] == "web_verified_dicom_uid"
    assert len({row["source_study_id"] for row in result["rows"]}) == 2
    assert client.get(f"/api/jobs/{response.json()['job_id']}/export.csv").status_code == 200


def test_singleton_mode_is_explicitly_preliminary(tmp_path):
    app = create_app(tmp_path, manager=_manager(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/api/jobs",
        data={"singleton_mode": "true"},
        files={"files": ("single.dcm", _dicom_bytes(generate_uid()), "application/dicom")},
    )
    status = _wait(client, response.json()["job_id"])
    assert status["status"] == "completed"
    result = client.get(f"/api/jobs/{response.json()['job_id']}/results").json()
    assert result["web_grouping"]["mode"] == "experimental_singleton_bag"
    assert result["web_export"]["singleton_export_allowed"] is False


def test_health_is_local_and_does_not_load_models(tmp_path):
    manager = _manager(tmp_path)
    client = TestClient(create_app(tmp_path, manager=manager))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["offline"] is True
    assert response.json()["runtime_loaded"] is False
