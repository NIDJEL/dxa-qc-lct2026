import json
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from dxa_qc.service import _hash_pixels, process_study


def _write_dicom(path: Path, study_uid: str, pixels: np.ndarray):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype(np.uint8).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path)


class _Runtime:
    device = "cpu"
    thresholds = np.full(10, 0.5)

    def __init__(self):
        self.calls = []

    def predict_bag(self, pixels):
        self.calls.append(list(pixels))
        return np.full(10, 0.1), 0.0


def test_one_study_with_multiple_series_uses_one_bag(tmp_path):
    study = generate_uid()
    _write_dicom(tmp_path / "series_a" / "a.dcm", study, np.zeros((20, 20)))
    _write_dicom(tmp_path / "series_b" / "b.dcm", study, np.ones((20, 20)))

    rt = _Runtime()
    result = process_study(tmp_path, runtime=rt)

    assert result["study_count"] == 1
    assert len(rt.calls) == 1
    assert len(rt.calls[0]) == 2
    assert {row["study_uid"] for row in result["rows"]} == {study}


def test_independent_studies_in_one_directory_are_separate_bags(tmp_path):
    first, second = generate_uid(), generate_uid()
    _write_dicom(tmp_path / "study_a" / "a.dcm", first, np.zeros((20, 20)))
    _write_dicom(tmp_path / "study_b" / "b.dcm", second, np.ones((20, 20)))
    mapping = tmp_path / "study_groups.json"
    mapping.write_text(json.dumps({
        "schema_version": "dxa_study_groups_v1",
        "studies": [
            {"source_study_id": "source-a", "files": ["study_a/a.dcm"]},
            {"source_study_id": "source-b", "files": ["study_b/b.dcm"]},
        ],
    }), encoding="utf-8")

    rt = _Runtime()
    result = process_study(tmp_path, runtime=rt, study_groups_path=mapping)

    assert result["study_count"] == 2
    assert len(rt.calls) == 2
    assert all(len(call) == 1 for call in rt.calls)
    assert {row["study_uid"] for row in result["rows"]} == {first, second}
    assert {row["source_study_id"] for row in result["rows"]} == {"source-a", "source-b"}


def test_exact_duplicate_dicoms_stay_in_one_study_bag(tmp_path):
    import shutil

    study = generate_uid()
    pixels = np.full((20, 20), 7, dtype=np.uint8)
    _write_dicom(tmp_path / "study" / "copy_a.dcm", study, pixels)
    shutil.copyfile(tmp_path / "study" / "copy_a.dcm", tmp_path / "study" / "copy_b.dcm")
    assert (tmp_path / "study" / "copy_a.dcm").read_bytes() == (tmp_path / "study" / "copy_b.dcm").read_bytes()

    rt = _Runtime()
    result = process_study(tmp_path, runtime=rt)

    assert result["study_count"] == 1
    assert len(rt.calls) == 1
    assert len(rt.calls[0]) == 2
    assert _hash_pixels(rt.calls[0][0]) == _hash_pixels(rt.calls[0][1])


def test_explicit_source_map_allows_different_study_uids_in_one_source_study(tmp_path):
    first, second = generate_uid(), generate_uid()
    _write_dicom(tmp_path / "same_source" / "spine.dcm", first, np.zeros((20, 20)))
    _write_dicom(tmp_path / "same_source" / "hip.dcm", second, np.ones((20, 20)))
    mapping = tmp_path / "study_groups.json"
    mapping.write_text(json.dumps({
        "schema_version": "dxa_study_groups_v1",
        "studies": [{
            "source_study_id": "verified-source-001",
            "files": ["same_source/spine.dcm", "same_source/hip.dcm"],
        }],
    }), encoding="utf-8")

    rt = _Runtime()
    result = process_study(tmp_path, runtime=rt, study_groups_path=mapping)

    assert result["grouping_method"] == "explicit_source_study"
    assert result["study_count"] == 1
    assert len(rt.calls) == 1
    assert {row["source_study_id"] for row in result["rows"]} == {"verified-source-001"}


def test_flat_multi_uid_input_is_separate_bags(tmp_path):
    _write_dicom(tmp_path / "a.dcm", generate_uid(), np.zeros((20, 20)))
    _write_dicom(tmp_path / "b.dcm", generate_uid(), np.ones((20, 20)))

    rt = _Runtime()
    result = process_study(tmp_path, runtime=rt)

    assert result["study_count"] == 2
    assert result["row_count"] == 2
    assert len(rt.calls) == 2


def test_different_uids_under_series_directories_are_separate(tmp_path):
    _write_dicom(tmp_path / "series_a" / "a.dcm", generate_uid(), np.zeros((20, 20)))
    _write_dicom(tmp_path / "series_b" / "b.dcm", generate_uid(), np.ones((20, 20)))
    rt = _Runtime()

    result = process_study(tmp_path, runtime=rt)

    assert result["study_count"] == 2
    assert len(rt.calls) == 2
    assert not any(e["error_type"] == "AmbiguousStudyGrouping" for e in result["errors"])
