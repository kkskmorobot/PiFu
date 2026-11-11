from pathlib import Path
import csv,json,hashlib,statistics,math
P=Path(r'E:\App\DaiMa\JetBrains\Project\PiFuV2')
O=P/'LunWen/astra-6-ultra/Data/academic_rewrite'
def j(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def c(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return [{k.strip():v for k,v in r.items()} for r in csv.DictReader(f)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def flat(p):return {l.split(':',1)[0].strip():l.split(':',1)[1].strip() for l in p.read_text().splitlines() if ':' in l and not l.startswith((' ','#'))}
M={'binary_iou':'metrics/binary_lesion_iou','binary_dice':'metrics/binary_lesion_dice','image_macro_f1':'metrics/image_macro_f1','group_macro_f1':'metrics/group_macro_f1','joint_score':'metrics/joint_score','mel_to_nv_rate':'metrics/group_mel_to_nv_rate','nll':'metrics/group_nll','brier':'metrics/group_brier','ece':'metrics/group_ece'}
records=[]
for r in c(P/'LunWen/Data/07_current_material_audit_20260916/run_summary.csv'):
 p=Path(r['source_results_csv']);rows=c(p);ap=p.parent/'checkpoint_bundles/aliases/best_joint_score.json';a=j(ap);ep=a['epoch']+1
 assert a['bundle']==r['bundle'] and ep==int(r['visible_epoch'])
 row=next(x for x in rows if int(float(x['epoch']))==ep);bp=p.parent/'checkpoint_bundles'/a['bundle'];man=j(bp/'manifest.json');assert man['epoch']==a['epoch']
 method='Plain' if r['run'].isnumeric() else ('LGLA' if r['run'] in ['lgx','lg2','lg3'] else {'lg1':'No balance','ab1':'Constant guidance','ab2':'Fixed equal fusion'}[r['run']])
 v={k:float(row[col]) for k,col in M.items()};assert abs(v['joint_score']-math.sqrt(v['binary_iou']*v['group_macro_f1']))<1e-5
 records.append(dict(method=method,run=r['run'],seed=int(r['seed']),epoch=ep,bundle=a['bundle'],fingerprint=man['fingerprint'],source=str(p),csv_sha256=sha(p),alias_source=str(ap),bundle_source=str(bp),training_rows=len(rows),metrics=v,curve=[{k:float(x[col]) for k,col in {'epoch':'epoch',**M}.items()} for x in rows]))
summary=[]
for m in M:
 pv=[r['metrics'][m] for r in records if r['method']=='Plain'];lv=[r['metrics'][m] for r in records if r['method']=='LGLA'];pm=statistics.mean(pv);lm=statistics.mean(lv)
 summary.append(dict(metric=m,plain_mean=pm,plain_sample_sd=statistics.stdev(pv),lgla_mean=lm,lgla_sample_sd=statistics.stdev(lv),absolute_delta=lm-pm,relative_gain_percent=(lm-pm)/pm*100))
attention=[];checks=[];base=next(r for r in records if r['run']=='18');ba=flat(Path(base['source']).parent/'args.yaml');bm=c(P/'tools/run18/run18_development_manifest.csv')
keys=['task','epochs','patience','batch','imgsz','optimizer','lr0','lrf','weight_decay','cos_lr','deterministic','amp','hsv_h','hsv_s','hsv_v','degrees','translate','scale','flipud','fliplr','mosaic','mixup','cutmix','copy_paste']
for r in c(P/'tools/sum1/runs.csv'):
 if r['family'] not in ['ECA','ECA+Spatial7','CBAM']:continue
 p=Path(r['source']);a=j(p.parent/'checkpoint_bundles/aliases/best_joint_score.json');ep=a['epoch']+1;assert ep==int(r['best_epoch']);row=next(x for x in c(p) if int(float(x['epoch']))==ep)
 attention.append(dict(method=r['family'],run=r['run'],seed=int(r['seed']),epoch=ep,bundle=a['bundle'],source=str(p),csv_sha256=sha(p),metrics={k:float(row[col]) for k,col in M.items()}))
 aa=flat(p.parent/'args.yaml');prefix='run'+r['run'];mp=P/'tools'/prefix/(prefix+'_development_manifest.csv');cfg=P/'tools'/prefix/(prefix+'_preregistered_config_v1.yaml')
 checks.append(dict(run=r['run'],args_source=str(p.parent/'args.yaml'),common_parameter_differences={k:[ba.get(k),aa.get(k)] for k in keys if ba.get(k)!=aa.get(k)},manifest_source=str(mp),manifest_sha256=sha(mp),manifest_rows_equal_run18=c(mp)==bm,config_source=str(cfg)))
tests={};teststats=[]
for t in ['test1','test2']:
 p=P/'experiments'/t/'results/result.json';d=j(p);tests[t]=dict(source=str(p),sha256=sha(p),data=d)
 for sec,path,ms in [('internal_segmentation',('internal_test','segmentation'),['binary_iou','binary_dice']),('external_segmentation',('external','segmentation'),['binary_iou','binary_dice']),('restricted_classification',('internal_test','classification'),['accuracy','mel_recall','nv_recall'])]:
  for m in ms:
   vals=[s[path[0]][path[1]][m] for s in d['per_seed']];mean=statistics.mean(vals);sd=statistics.stdev(vals)
   assert abs(mean-d['aggregate'][sec][m]['mean'])<1e-10 and abs(sd-d['aggregate'][sec][m]['sample_sd'])<1e-10
   teststats.append(dict(test=t,section=sec,metric=m,values=vals,mean=mean,sample_sd=sd))
 for s in d['per_seed']:
  cp=P/'experiments'/t/'results'/('seed'+str(s['seed']))/'internal/classification.csv';rr=c(cp);assert len(rr)==56 and sum(x['argmax_class']==x['true_class'] for x in rr)/56==s['internal_test']['classification']['accuracy']
  s['saved_csv_confusion']={y:{pred:sum(x['true_class']==y and x['argmax_class']==pred for x in rr) for pred in ['akiec','bcc','bkl','df','mel','nv','vasc','scc']} for y in ['mel','nv']};s['classification_csv_source']=str(cp)
perf=P/'LunWen/Data/07_current_material_audit_20260916/supplements_20260916/performance/unified_performance_results.json';qual=P/'tools/sum2/attention_heatmaps_seed43.json'
data=dict(verification_date='2026-11-09',development=records,summary=summary,historical_attention=attention,attention_config_checks=checks,test_results=tests,test_recomputed=teststats,test_manifest_hashes={t:{n:sha(P/'tools'/t/n) for n in ['manifest_test.csv','manifest_external.csv']} for t in ['test1','test2']},performance=dict(source=str(perf),sha256=sha(perf),data=j(perf)),qualitative=dict(source=str(qual),data=j(qual),figure=str(P/'tools/sum2/figure_attention_heatmaps_seed43.png'),script=str(P/'tools/sum2/make_attention_heatmaps.py')),limits=['No model inference or new checkpoint selection.','Development metrics use original CSV precision.','No new binary checkpoint hash audit.','Diagnostic 1746 forwards are not official 1744-image validation metrics.'])
(O/'results_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
fields=['method','run','seed','dataset','checkpoint','metric','raw_value','aggregation','absolute_delta','relative_gain_percent','source']
with (O/'实验汇总与指标换算.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in records+attention:
  for m,v in r['metrics'].items():w.writerow(dict(method=r['method'],run=r['run'],seed=r['seed'],dataset='development validation',checkpoint=r['bundle'],metric=m,raw_value=v,aggregation='single retained CSV row',source=r['source']))
 for r in summary:
  for family in ['plain','lgla']:
   w.writerow(dict(method=family.upper(),run='3 seeds',seed='42/43/44',dataset='development validation',checkpoint='retained best Joint per run',metric=r['metric'],raw_value=r[family+'_mean'],aggregation='mean; sample SD='+str(r[family+'_sample_sd']),absolute_delta=r['absolute_delta'] if family=='lgla' else '',relative_gain_percent=r['relative_gain_percent'] if family=='lgla' else '',source=';'.join(x['source'] for x in records if x['method'].lower()==family)))
 for r in teststats:w.writerow(dict(method='Plain' if r['test']=='test1' else 'LGLA',run=r['test'],seed='42/43/44',dataset=r['section'],checkpoint='frozen per seed',metric=r['metric'],raw_value=r['mean'],aggregation='mean; sample SD='+str(r['sample_sd']),source=tests[r['test']]['source']))
print('Verified 9 primary runs, 7 attention runs, 2 fixed-test result sets; wrote results_data.json and metric CSV.')
