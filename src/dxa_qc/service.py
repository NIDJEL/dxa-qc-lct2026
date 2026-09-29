"""Operational local DXA QC service using frozen DINOv3, MedImageInsight and E007 A3.
E007 targets are study-level proxies and are never reported as official image-level metrics.
"""
from __future__ import annotations
import csv, hashlib, json, time
from pathlib import Path
import numpy as np

VIOLATIONS=("spine_positioning","spine_axis","spine_foreign_object","femur_positioning_rotation","femur_roi")
RU_REGION={"spine":"Поясничный отдел позвоночника","hip":"Проксимальный отдел бедра","unknown":"Неизвестно"}
RU_VIOLATION={"spine_positioning":"Некорректная укладка","spine_axis":"Не выравнена ось позвоночника","spine_foreign_object":"Присутствуют посторонние предметы","femur_positioning_rotation":"Некорректная укладка","femur_roi":"Некорректная область интереса"}

def _hash_pixels(a):
    h=hashlib.sha256(); h.update(f"{a.shape}|{a.dtype}|".encode()); h.update(np.ascontiguousarray(a).tobytes()); return h.hexdigest()
def load_manifest(path):
    p=Path(path) if path else None
    return {r["pixel_hash"]:r for r in json.loads(p.read_text(encoding="utf-8"))} if p and p.exists() else {}
def _read(path):
    import pydicom
    ds=pydicom.dcmread(str(path),force=False)
    arr=np.asarray(ds.pixel_array)
    if arr.ndim != 2 or arr.dtype != np.uint8 or ds.PhotometricInterpretation != "MONOCHROME2":
        raise ValueError("E007 supports audited uint8 MONOCHROME2 single-frame DICOM only")
    return ds,arr
def _geometry(ds,arr):
    out={"status":"unknown","pixel_rows":int(arr.shape[-2]),"pixel_columns":int(arr.shape[-1]),"row_spacing_mm":None,"column_spacing_mm":None,"physical_extent_mm":None}
    try:
        r,c=map(float,getattr(ds,"PixelSpacing",[]))
        if r>0 and c>0: out.update(status="valid_dicom_pixel_spacing",row_spacing_mm=r,column_spacing_mm=c,physical_extent_mm={"rows":arr.shape[-2]*r,"columns":arr.shape[-1]*c})
    except (TypeError,ValueError): pass
    return out
def _content_route(arr,rec=None):
    """Route from pixels only; manifest/study labels are never router inputs."""
    x=arr.astype(np.float32); mask=x>np.percentile(x,18); ys,xs=np.where(mask)
    if len(xs)<100:return "unknown","content_abstain"
    spread=(xs.max()-xs.min())/max(ys.max()-ys.min(),1)
    if spread>=1.15:return "hip","content_heuristic"
    return "unknown","content_abstain"

class Runtime:
    def __init__(self,root,checkpoint,device="auto"):
        import torch
        from .features import load_encoder,encoder_forward
        from .model import E007
        self.torch=torch; self.root=Path(root); self.encoder_forward=encoder_forward
        self.device=torch.device("cuda" if device=="auto" and torch.cuda.is_available() else (device if device!="auto" else "cpu"))
        checkpoint=Path(checkpoint)
        if not checkpoint.is_absolute(): checkpoint=self.root/checkpoint
        ck=torch.load(checkpoint,map_location=self.device,weights_only=True); self.checkpoint=ck
        self.provenance={"checkpoint":str(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         "checkpoint_provenance":ck["provenance"],
                         "thresholds":np.asarray(ck["calibration"]["threshold"]).tolist(),
                         "temperatures":np.asarray(ck["calibration"]["temperature"]).tolist(),
                         "dino_preprocessing":"448 letterbox","mi2_preprocessing":"native 480 square"}
        from . import TARGETS
        if ck["provenance"]["ablation"] != "A3" or ck["provenance"]["target_order"] != list(TARGETS):
            raise ValueError("Expected E007 A3 checkpoint with canonical target order")
        self.model=E007(ck["provenance"]["ablation"]).to(self.device); self.model.load_state_dict(ck["model"],strict=True); self.model.eval()
        self.thresholds=np.asarray(ck["calibration"]["threshold"],dtype=np.float64)
        self.temperature=np.asarray(ck["calibration"]["temperature"], dtype=np.float64)
        if self.temperature.shape != (10,) or not np.isfinite(self.temperature).all() or (self.temperature<=0).any():
            raise ValueError("Invalid checkpoint temperature calibration")
        if self.thresholds.shape != (10,) or not np.isfinite(self.thresholds).all() or ((self.thresholds<0)|(self.thresholds>1)).any():
            raise ValueError("Invalid checkpoint threshold calibration")
        self.dino=load_encoder(self.root,"dino",self.device); self.mi2=load_encoder(self.root,"mi2",self.device)
        router_path=self.root/"artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL/router_production.json"
        self.router=json.loads(router_path.read_text(encoding="utf-8"))
        if (self.router["schema_version"]!="E023_anatomy_router_v1"
                or self.router["feature_order"]!=["dino_l2_0:1024","mi2_l2_0:1024"]
                or len(self.router["models"])!=3):
            raise ValueError("Invalid frozen anatomy router")
        self.provenance["router_sha256"]=hashlib.sha256(router_path.read_bytes()).hexdigest()
        self.provenance["router_scope"]="provisional visual anatomy, not expert validation"
    def predict(self,pixels,ds=None):
        return self.predict_bag([pixels])
    def extract(self, pixels):
        return {"dino":self.encoder_forward(self.dino,"dino",pixels,self.device)["dino"],
                "mi2":self.encoder_forward(self.mi2,"mi2",pixels,self.device)["mi2"]}
    def predict_features(self, features):
        import torch
        from scipy.special import expit
        batch={k:torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in features.items()}
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            output=self.model(batch)
        logits=output["logits"].float().cpu().numpy()
        if logits.shape != (10,) or not np.isfinite(logits).all():
            raise ValueError("Expected ten finite study logits")
        self.last_trace={"logits":logits,
                          "gates":output["gates"].float().cpu().numpy(),
                          "attention":{k:v.float().cpu().numpy() for k,v in output["attention"].items()}}
        return expit(logits/self.temperature)
    def predict_bag(self,pixels_list):
        t=time.perf_counter()
        unique={_hash_pixels(x):x for x in pixels_list}
        features=[self.extract(unique[k]) for k in sorted(unique)]
        p=self.predict_features({k:np.stack([r[k] for r in features]) for k in ("dino","mi2")})
        return p,time.perf_counter()-t
    def route_features(self,features):
        branches=[]
        for key in ("dino","mi2"):
            v=np.asarray(features[key],dtype=np.float64)
            if v.shape!=(1024,) or not np.isfinite(v).all() or np.linalg.norm(v)==0:
                return "unknown","linear_router_invalid_features",None
            branches.append(v/np.linalg.norm(v))
        x=np.concatenate(branches)
        probabilities=[]
        for m in self.router["models"]:
            z=np.dot((x-np.asarray(m["mean"]))/np.asarray(m["scale"]),np.asarray(m["coef"]))+m["intercept"]
            probabilities.append(float(1/(1+np.exp(-np.clip(z,-100,100)))))
        p=float(np.mean(probabilities))
        margin=float(self.router["abstention_margin"])
        region="spine" if p>=.5+margin else "hip" if p<.5-margin else "unknown"
        return region,"frozen_linear_router" if region!="unknown" else "linear_router_abstain",p
    def predict_bag_routed(self,pixels_list):
        t=time.perf_counter()
        unique={_hash_pixels(x):x for x in pixels_list}
        keys=sorted(unique)
        features=[self.extract(unique[k]) for k in keys]
        routes={k:self.route_features(f) for k,f in zip(keys,features)}
        p=self.predict_features({k:np.stack([r[k] for r in features]) for k in ("dino","mi2")})
        return p,time.perf_counter()-t,routes

def _pred(p,region):
    if region=="spine": return p[[0,1,2]],p[7]
    if region=="hip": return np.array([max(p[3],p[5]),max(p[4],p[6])]),max(p[8],p[9])
    return np.array([p[0],p[1],p[2],max(p[3],p[5]),max(p[4],p[6])]),max(p[7],p[8],p[9])

def _study_groups(rootpath, files, records, study_groups_path=None):
    """Resolve bag boundaries without using source labels or anatomy as evidence.

    An explicit, independently verified file-to-study map can override rewritten
    DICOM UIDs. Without one, only consistent nonempty StudyInstanceUIDs are used.
    """
    valid = {str(r[0]): r for r in records}
    if study_groups_path:
        spec = json.loads(Path(study_groups_path).read_text(encoding="utf-8"))
        if spec.get("schema_version") != "dxa_study_groups_v1" or not isinstance(spec.get("studies"), list):
            raise ValueError("Invalid study-groups schema")
        groups, seen = {}, set()
        for entry in spec["studies"]:
            sid = entry.get("source_study_id")
            names = entry.get("files")
            if not isinstance(sid, str) or not sid.strip() or not isinstance(names, list) or not names:
                raise ValueError("Each declared study needs a nonempty source_study_id and files")
            if sid in groups:
                raise ValueError(f"Duplicate declared source_study_id: {sid}")
            paths = []
            for name in names:
                if not isinstance(name, str) or not name:
                    raise ValueError("Invalid declared DICOM path")
                f = rootpath / name if rootpath.is_dir() else rootpath.parent / name
                if f not in files or str(f) in seen:
                    raise ValueError(f"Missing, repeated or out-of-input study file: {name}")
                seen.add(str(f))
                paths.append(str(f))
            groups[sid] = [valid[p] for p in paths if p in valid]
        if seen != {str(f) for f in files}:
            raise ValueError("Study-groups map does not cover every input DICOM")
    else:
        by_path = {str(r[0]): r for r in records}
        def uid_set(items):
            return {str(getattr(by_path[str(f)][1], "StudyInstanceUID", "")).strip()
                    for f in items if str(f) in by_path}
        def subtree(folder):
            return sorted(folder.rglob("*.dcm"))

        if not rootpath.is_dir():
            groups = {"input_file:" + rootpath.name: list(records)}
        else:
            direct = sorted(rootpath.glob("*.dcm"))
            children = [p for p in sorted(rootpath.iterdir()) if p.is_dir() and subtree(p)]
            if not direct and not children:
                raise ValueError("No DICOM files found")
            else:
                groups = {}
                for f in files:
                    record = by_path.get(str(f))
                    if record is None:
                        continue
                    uid = str(getattr(record[1], "StudyInstanceUID", "") or "").strip()
                    series = str(getattr(record[1], "SeriesInstanceUID", "") or "").strip()
                    if not uid:
                        if len(files) == 1:
                            uid = "input_file:" + f.name
                        else:
                            raise ValueError("Missing StudyInstanceUID in multi-file input; explicit study-groups map required")
                    groups.setdefault("dicom_uid:" + uid, []).append(record)
                series_owner = {}
                for sid, bag in groups.items():
                    for record in bag:
                        series = str(getattr(record[1], "SeriesInstanceUID", "") or "").strip()
                        if not series:
                            continue
                        prior = series_owner.setdefault(series, sid)
                        if prior != sid:
                            raise ValueError("SeriesInstanceUID crosses StudyInstanceUID boundaries")
    # Identical decoded images must never land in independent study bags.
    owners = {}
    for sid, bag in groups.items():
        for r in bag:
            h = _hash_pixels(r[2])
            if h in owners and owners[h] != sid:
                raise ValueError("Identical decoded DICOM pixels cross study boundaries; resolve grouping explicitly")
            owners[h] = sid
    return groups

def _web_uid_groups(records):
    """Web-only partial grouping from verified Study/Series UIDs.

    Unlike the fail-closed competition CLI, independent unambiguous studies
    may proceed when an unrelated input file/group is invalid. Never use
    folder names, rewritten UID as patient identity, or anatomy as evidence.
    """
    groups, errors, series_owner, pixel_owner = {}, [], {}, {}
    for record in records:
        f, ds, arr = record[:3]
        uid = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
        series = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if not uid or not series:
            errors.append({"image_path": str(f), "error_type": "AmbiguousStudyGrouping",
                           "error": "Missing StudyInstanceUID or SeriesInstanceUID; isolated from other studies"})
            continue
        owner = series_owner.setdefault(series, uid)
        if owner != uid:
            errors.append({"image_path": str(f), "error_type": "AmbiguousStudyGrouping",
                           "error": "SeriesInstanceUID crosses StudyInstanceUID boundaries"})
            continue
        key = "dicom_uid:" + uid
        groups.setdefault(key, []).append(record)
        pixel_owner.setdefault(_hash_pixels(arr), set()).add(key)
    crossed = {g for owners in pixel_owner.values() if len(owners) > 1 for g in owners}
    for key in crossed:
        errors.extend({"image_path": str(r[0]), "error_type": "AmbiguousStudyGrouping",
                       "error": "Identical decoded pixels cross study boundaries"}
                      for r in groups.pop(key))
    return groups, errors


def _automatic_uid_groups(records):
    """Keep independently identified studies when another record is ambiguous."""
    groups, errors, series_owner, pixel_owner = {}, [], {}, {}
    single = len(records) == 1
    for record in records:
        f, ds, arr = record[:3]
        uid = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
        if not uid:
            if single:
                uid = "single_input:" + Path(f).name
            else:
                errors.append({"image_path": str(f), "error_type": "AmbiguousStudyGrouping",
                               "error": "Missing StudyInstanceUID in multi-file input"})
                continue
        key = "dicom_uid:" + uid
        groups.setdefault(key, []).append(record)
        series = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if series:
            series_owner.setdefault(series, set()).add(key)
        pixel_owner.setdefault(_hash_pixels(arr), set()).add(key)
    invalid = {key for owners in series_owner.values() if len(owners) > 1 for key in owners}
    invalid.update(key for owners in pixel_owner.values() if len(owners) > 1 for key in owners)
    for key in invalid:
        errors.extend({"image_path": str(r[0]), "error_type": "AmbiguousStudyGrouping",
                       "error": "Series UID or identical decoded pixels cross study boundaries"}
                      for r in groups.pop(key, []))
    return groups, errors


def process_study(path,manifest_path=None,output_path=None,runtime=None,checkpoint=None,root=None,device="auto",study_groups_path=None,
                  web_auto_groups=False, progress_callback=None):
    rootpath=Path(path); files=sorted(rootpath.rglob("*.dcm")) if rootpath.is_dir() else [rootpath]
    # Optional historical manifest is deliberately never read at inference.
    root = root or Path(__file__).parents[2]; checkpoint = checkpoint or root / "artifacts/E007_FULL_V1/A3/seed_17/fold_0/checkpoints/best.pt"
    start=time.perf_counter(); rows=[]; errors=[]; model_times=[]; records=[]
    for f in files:
        t=time.perf_counter()
        try:
            ds,arr=_read(f); region,route="unknown","pending_frozen_router"
            records.append((f,ds,arr,region,route,t))
        except Exception as e: errors.append({"image_path":str(f),"error_type":type(e).__name__,"error":str(e)})
    if not study_groups_path:
        groups, grouping_errors = _automatic_uid_groups(records)
        errors.extend(grouping_errors)
    else:
        try:
            groups = _study_groups(rootpath, files, records, study_groups_path)
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as e:
            groups = {}
            errors.extend({"image_path": str(r[0]), "error_type": "AmbiguousStudyGrouping", "error": str(e)} for r in records)
    if progress_callback:
        progress_callback("dicom_read", len(records), len(files))
    if groups and runtime is None:
        runtime = Runtime(root, checkpoint, device)
    predictions={}
    routing={}
    for group_index, (key, bag) in enumerate(groups.items(), 1):
        try:
            if hasattr(runtime,"predict_bag_routed"):
                p,mt,routes=runtime.predict_bag_routed([r[2] for r in bag])
                for r in bag:
                    routing[str(r[0])]=routes[_hash_pixels(r[2])]
            else:
                p,mt=runtime.predict_bag([r[2] for r in bag])
                for r in bag:
                    region,route=_content_route(r[2])
                    routing[str(r[0])]=(region,route,None)
            model_times.append(mt)
            if np.asarray(p).shape != (10,) or not np.isfinite(p).all():
                raise ValueError("Expected one finite ten-target vector per source study")
            for r in bag: predictions[str(r[0])]=(p,key)
        except Exception as e:
            errors.extend({"image_path":str(r[0]),"error_type":type(e).__name__,"error":str(e)} for r in bag)
        if progress_callback:
            progress_callback("study_bag", group_index, len(groups))
    for f,ds,arr,region,route,t in records:
        if str(f) not in predictions: continue
        probs_row,source_study_id=predictions[str(f)]
        try:
            region,route,router_probability=routing[str(f)]
            vals,q=_pred(probs_row,region); q=float(np.clip(q,0,1)) if np.isfinite(q) else None
            suid=str(getattr(ds,"StudyInstanceUID","unknown")); iuid=str(getattr(ds,"SOPInstanceUID",f.name))
            applicable=VIOLATIONS[:3] if region=="spine" else (VIOLATIONS[3:] if region=="hip" else VIOLATIONS)
            probs=[(RU_VIOLATION[v],float(prob)) for v,prob in zip(applicable,vals)]
            flags=probs_row>=runtime.thresholds
            indices=([0],[1],[2]) if region=="spine" else (([3,5],[4,6]) if region=="hip" else ([0],[1],[2],[3,5],[4,6]))
            violation_names=[name for (name,_),ii in zip(probs,indices) if flags[ii].any()]
            quality_indices=[7] if region=="spine" else ([8,9] if region=="hip" else [7,8,9])
            rows.append({"path_to_study":str(rootpath),"study_uid":suid,"image_uid":iuid,"image_path":str(f),"anatomical_region":RU_REGION[region] if region!="unknown" else "","hip_side":"unknown","quality_class":int(flags[quality_indices].any()) if region!="unknown" and q is not None else None,"quality_prob":q,"quality_target":"quality_spine" if region=="spine" else ("quality_right_hip_or_left_hip_max" if region=="hip" else None),"violation_type":";".join(violation_names) if region!="unknown" else "","violation_probability":max(x[1] for x in probs),"violation_status":"predicted_proxy" if region!="unknown" else "route_abstained","routing_status":route,"router_p_spine":router_probability,"source_study_id":source_study_id,"study_proxy_probabilities":probs_row.tolist(),"head_target_order":["spine_positioning","spine_axis","spine_foreign_object","right_hip_positioning_rotation","right_hip_roi","left_hip_positioning_rotation","left_hip_roi","quality_spine","quality_right_hip","quality_left_hip"],"head_thresholds":runtime.thresholds.tolist(),"geometry":_geometry(ds,arr),"processing_status":"Success" if region!="unknown" else "Failure","time_of_processing":time.perf_counter()-t})
            if region=="unknown":
                errors.append({"image_path":str(f),"error_type":"UnknownAnatomy","error":"Pixel-only anatomy routing abstained; no official label emitted"})
        except Exception as e: errors.append({"image_path":str(f),"error_type":type(e).__name__,"error":str(e)})
    result={"schema_version":"dxa_qc_pdf_recommended_eight_fields_unconfirmed","scope":"operational_E007_A3_study_proxy","study_path":str(rootpath),"image_count":len(files),"study_count":len([g for g in groups.values() if g]),"grouping_method":"explicit_source_study" if study_groups_path else ("web_verified_dicom_uid" if web_auto_groups else "dicom_study_uid"),"row_count":len(rows),"error_count":len(errors),"processing_status":"ok" if not errors else "partial_failure","elapsed_seconds":time.perf_counter()-start,"model_seconds":model_times,"device":str(runtime.device) if runtime else "not_loaded","provenance":getattr(runtime,"provenance",{"runtime":"not_loaded" if runtime is None else "injected"}),"rows":rows,"errors":errors,"limitations":["E007 targets are study-level proxies, not official image-level metrics","hip laterality is unknown; side-specific heads are combined by max for review","no expert image-level calibration or landmark model; unknown routes remain null","DINOv3 and MedImageInsight run locally"]}
    if output_path: Path(output_path).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    return result

def write_official_csv(result,path):
    from .contract import CSV_FIELDS
    fields=CSV_FIELDS
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k) for k in fields} for r in result["rows"])
