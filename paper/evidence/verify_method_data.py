from pathlib import Path
import csv,json,hashlib,pickletools,zipfile,collections,re
R=Path(r'E:\App\DaiMa\JetBrains\Project\PiFuV2')
O=R/'LunWen/astra-6-ultra/Data/academic_rewrite'
def rows(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
manifest=R/'tools/lgx/lgx_development_manifest.csv'
a=rows(manifest)
splits={}
for s in ['train','val']:
 x=[v for v in a if v['split']==s]
 splits[s]={'total':len(x),'seg':sum(v['has_segmentation']=='1' for v in x),'cls':sum(v['has_classification']=='1' for v in x),'both':sum(v['has_segmentation']==v['has_classification']=='1' for v in x),'class_counts':dict(collections.Counter(v['class_name'] for v in x if v['has_classification']=='1')),'dataset_role_counts':dict(collections.Counter(v['dataset_role'] for v in x)),'empty_group_id':sum(not v['group_id'].strip() for v in x),'classification_groups':len({v['group_id'] or v['isic_id'] for v in x if v['has_classification']=='1'}),'empty_lesion_id':sum(not v['lesion_id'].strip() for v in x),'empty_patient_id':sum(not v['patient_id'].strip() for v in x)}
overlap={}
for key in ['isic_id','group_id','image_sha256','lesion_id','patient_id']:
 ss=[{v[key].strip() for v in a if v['split']==s and v[key].strip()} for s in ['train','val']]
 overlap[key]=len(ss[0]&ss[1])
pre=json.loads((R/'tools/lgx/LGX_PREFLIGHT_REPORT.json').read_text(encoding='utf8'))
hashcheck=[]
for p,info in pre['lock_inputs'].items():
 actual=sha(Path(p));hashcheck.append({'path':p,'expected':info['sha256'],'actual':actual,'match':actual==info['sha256']})
runrows=rows(R/'LunWen/Data/07_current_material_audit_20260916/run_summary.csv')
run_data=[]
for rr in runrows:
 cp=Path(rr['source_results_csv']); args=cp.parent/'args.yaml'
 d={}
 for line in args.read_text(encoding='utf8').splitlines():
  if ':' in line:
   k,v=line.split(':',1);d[k]=v.strip()
 b=cp.parent/'checkpoint_bundles'/rr['bundle']/'manifest.json'
 bi=json.loads(b.read_text(encoding='utf8'))
 run_data.append({'run':rr['run'],'seed':int(rr['seed']),'args_path':str(args),'args_sha256':sha(args),'arguments':d,'checkpoint_manifest':str(b),'epoch_zero':bi['epoch'],'fingerprint':bi['fingerprint']})
# Inspect pickle opcodes only; do not load models or execute pickle bytecode.
def checkpoint_tokens(path,keys):
 with zipfile.ZipFile(path) as z:
  name=next(n for n in z.namelist() if n.endswith('data.pkl'))
  ops=list(pickletools.genops(z.read(name)))
 out=[]
 for i,(op,arg,pos) in enumerate(ops):
  if isinstance(arg,str) and arg in keys:
   out.append({'key':arg,'offset':pos,'following_opcodes':[(o.name,v) for o,v,_ in ops[i:i+16]]})
 return out
ck=R/'experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43/checkpoint_bundles/epoch_0098/trainer.pt'
metadata=checkpoint_tokens(ck,{'scale_channels','version','git','date'})
# Existing data split records only, no image access or re-evaluation.
ima=rows(R/'Data/processed_imaplusplus_binary_v1/manifest.csv')
inc=rows(R/'Data/processed_isic2019_classification_incremental_v1/manifest.csv')
out={'verified_on':'2026-11-09','manifest':str(manifest),'manifest_sha256':sha(manifest),'development':splits,'train_val_nonempty_key_overlap':overlap,'ima_frozen_counts':dict(collections.Counter(v['split'] for v in ima)),'isic2019_frozen_counts':dict(collections.Counter(v['split'] for v in inc)),'lock_input_hash_check':hashcheck,'runs':run_data,'lgx_checkpoint_opcode_metadata':metadata,'pretrained_checkpoint_opcode_metadata':checkpoint_tokens(R/'models/yolo26m-sem.pt',{'version','date'}),'model_dimensions':{'P3':[256,80,80],'P4':[512,40,40],'main':[1,80,80],'aux_train':[1,40,40],'projection_each':128,'global_descriptor':256,'local_descriptor':256,'gate':[512,128,2],'classifier':[256,256,8],'dropout':0.2},'known_limits':['Ultralytics package git commit absent in checkpoint metadata. Core project source/config hashes verified against frozen preflight; installed dependency implementation read and version matches checkpoint, but dependency files were not included in original training fingerprint.','Split overlap checks apply to available metadata identifiers and frozen SHA256 fields; missing patient/lesion metadata does not establish patient independence.']}
(O/'method_data.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'counts':splits,'overlap':overlap,'hash_inputs':len(hashcheck),'hash_pass':sum(x['match'] for x in hashcheck),'runs':len(run_data)},ensure_ascii=False,indent=2))

# Verify frozen inputs for all nine principal runs, reusing hashes of shared sources.
all_checks={}; cache={}
for run in ['lg1','lgx','lg2','lg3','ab1','ab2','run18','run22','run24']:
 f=R/'tools'/run/(run.upper()+'_PREFLIGHT_REPORT.json'); raw=json.loads(f.read_text(encoding='utf8')); checks=[]
 for source,info in raw.get('lock_inputs',{}).items():
  q=Path(source)
  if not q.is_absolute():q=R/q
  if str(q) not in cache:cache[str(q)]=sha(q) if q.exists() else None
  checks.append({'path':str(q),'expected':info['sha256'],'actual':cache[str(q)],'match':cache[str(q)]==info['sha256']})
 all_checks[run]={'preflight':str(f),'fingerprint':raw.get('fingerprint'),'checks':checks}
out['all_run_input_hash_verification']=all_checks
(O/'method_data.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
