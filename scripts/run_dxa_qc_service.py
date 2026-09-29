import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import argparse,json
import torch
torch.set_num_threads(2)
from dxa_qc.service import process_study,write_official_csv
p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("--root",default=str(Path(__file__).resolve().parents[1])); p.add_argument("--checkpoint",default="artifacts/E007_FULL_V1/A3/seed_17/fold_0/checkpoints/best.pt"); p.add_argument("--manifest",default=None,help="Deprecated: ignored to prevent source-label leakage"); p.add_argument("--study-groups",default=None,help="Optional verified JSON file-to-study map for ambiguous archives"); p.add_argument("--device",default="auto"); p.add_argument("--json",default="dxa_qc_result.json"); p.add_argument("--csv",default="dxa_qc_result.csv"); a=p.parse_args(); r=process_study(a.input,a.manifest,a.json,checkpoint=a.checkpoint,root=a.root,device=a.device,study_groups_path=a.study_groups); write_official_csv(r,a.csv); print(json.dumps({k:r[k] for k in ("processing_status","image_count","study_count","row_count","error_count","elapsed_seconds","device")},ensure_ascii=False))

if r["error_count"] or not r["image_count"]: sys.exit(1)
