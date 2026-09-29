"""Local inference-only latency and resource measurement, no cached embeddings."""
import argparse,json,sys,time,threading,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from dxa_qc.service import Runtime,process_study,write_official_csv
from dxa_qc.contract import validate_csv
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--root',default=str(ROOT));a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
start=time.perf_counter();rt=Runtime(Path(a.root),Path(a.root)/'artifacts/E007_FULL_V1/A3/seed_17/fold_0/checkpoints/best.pt',a.device);load=time.perf_counter()-start
runs=[]
for mode in ('cold','warm'):
 t=time.perf_counter();r=process_study(a.input,runtime=rt);write_official_csv(r,out/(mode+'.csv'));elapsed=time.perf_counter()-t
 assert r['error_count']==0 and r['row_count']==r['image_count'] and r['image_count']>0,r['errors']
 assert not validate_csv(out/(mode+'.csv'))
 (out/(mode+'.json')).write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf8')
 runs.append({'mode':mode,'wall_seconds':elapsed,'with_model_load_seconds':elapsed+load if mode=='cold' else elapsed,'images':r['image_count'],'source_studies':len({x['source_study_id'] for x in r['rows']})})
if sys.platform=='linux':
 import resource
 ram=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
else:
 import ctypes
 from ctypes import wintypes
 class Counters(ctypes.Structure):
  _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
 getprocess=ctypes.windll.kernel32.GetCurrentProcess;getprocess.restype=wintypes.HANDLE
 getmem=ctypes.windll.psapi.GetProcessMemoryInfo;getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
 c=Counters();c.cb=ctypes.sizeof(c)
 if not getmem(getprocess(),ctypes.byref(c),c.cb):raise ctypes.WinError()
 ram=c.PeakWorkingSetSize

result={'platform':sys.platform,'device':str(rt.device),'model_load_seconds':load,'runs':runs,'peak_process_rss_bytes':ram,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if rt.device.type=='cuda' else None,'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if rt.device.type=='cuda' else None,'scope':'format-test smoke; source folder is explicit input bag; no model selection'}
(out/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
