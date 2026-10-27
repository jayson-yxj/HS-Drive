"""Offline fixed A/B/C aggregates. Never used for training/selection decisions."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m8j8_report import read,load,stripped,metrics,paired,write,sha
from training.m8j7_metrics import match,distribution


def paired_classes(a,b,classes):
    out={}
    for ci,c in enumerate(classes):
        def subset(records):
            return [dict(r,gt=r['gt'][r['gt_labels']==ci],gt_labels=r['gt_labels'][r['gt_labels']==ci]) for r in records]
        out[c]=paired(subset(a),subset(b),classes)
    return out


def anomalies(records,classes):
    counts=Counter();perclass=defaultdict(Counter)
    for r in records:
        for i,j in match(**stripped(r))['pairs']:
            p,g=r['boxes'][i],r['gt'][j]
            dz=abs(float(p[2]+p[5]/2-g[2]-g[5]/2));dh=abs(float(p[5]-g[5]))
            name=classes[int(r['gt_labels'][j])]
            counts['matched']+=1;perclass[name]['matched']+=1
            for t in (1.,2.):
                for key,v in [('abs_dz',dz),('abs_dh',dh)]:
                    k=f'{key}_gt{t:g}m';counts[k]+=int(v>t);perclass[name][k]+=int(v>t)
    return dict(scope='MatchedGTRecords_DescriptiveThresholds_NotSafetyGates',counts=dict(counts),per_class={c:dict(perclass[c]) for c in classes})


def main():
    cfg=read('configs/m8j10_protocol.json');private=Path(cfg['private_root']);rep=Path(os.environ['M8J10_REPORT']);source=rep.parent
    fixed=read(cfg['evaluation_manifest']);frames=fixed['frames'];classes=cfg['official_classes']
    train=read(source/'train/result.json');verify=read(source/'verify/result.json')
    assert train['status']=='Completed400Steps' and train['optimizer_steps']==400 and verify['status']=='FinalReloadPassed'
    steps=[json.loads(s) for s in (source/'train/steps.jsonl').read_text().splitlines()]
    order=read(cfg['manifest'])['order'];prepared=read(private/'prepared/manifest.json')
    assert len(steps)==len(order)==400
    for row,planned in zip(steps,order):
        assert row['global_step']==planned['step'] and row['frame_index']==planned['index'] and row['scene_name']==planned['scene_name']
        assert row['actual_lr']==1e-5 and row['optimizer_after']['weight_decay']==[0.] and row['frozen_parameter_and_buffer_max_abs_change']==0.
    exposure=Counter(r['sample_token'] for r in steps);scenes=sorted({f['scene_name'] for f in prepared['frames']})
    exp=dict(actual_optimizer_steps=400,unique_exposed_samples=len(exposure),pool_frames=len(prepared['frames']),
        per_scene={s:dict(steps=sum(r['scene_name']==s for r in steps),unique_exposed_samples=len({r['sample_token'] for r in steps if r['scene_name']==s})) for s in scenes},
        per_frame=[dict(index=f['index'],sample_token=f['sample_token'],scene_name=f['scene_name'],exposures=exposure[f['sample_token']]) for f in prepared['frames']],
        exposure_histogram=dict(Counter(exposure.values())),learning_rates=sorted({r['actual_lr'] for r in steps}),
        original_eight_exposures={f['sample_token']:exposure[f['sample_token']] for f in read('configs/m8j7_protocol.json')['frames']})
    write('reports/m8j10_exposure.json',exp)
    root8=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab')
    roots={'A':root8/'A/mini_val','B':root8/'B/mini_val','C200':private/'C200/mini_val','C400':private/'C400/mini_val'}
    records={n:[load(root/f'{f["index"]:02}.npz') for f in frames] for n,root in roots.items()}
    for n,rs in records.items():
        assert len(rs)==81 and sum(len(r['gt']) for r in rs)==3559
        for a,r in zip(records['A'],rs):
            np.testing.assert_array_equal(a['gt'],r['gt']);np.testing.assert_array_equal(a['gt_labels'],r['gt_labels'])
    agg={n:metrics(rs,classes,cfg) for n,rs in records.items()}
    old=read('reports/m8j8_ab_statistics.json')
    for n in ('A','B'):assert agg[n]==old['aggregate'][n], 'HistoricalProjectDiagnosticChanged'
    scene={}
    for s in sorted({f['scene_name'] for f in frames}):
        ids=[f['index'] for f in frames if f['scene_name']==s]
        subset={n:[rs[i] for i in ids] for n,rs in records.items()}
        scene[s]=dict(frames=len(ids),models={n:metrics(rs,classes,cfg) for n,rs in subset.items()},
            C400_vs_A=paired(subset['A'],subset['C400'],classes),C400_vs_B=paired(subset['B'],subset['C400'],classes))
    comparisons={}
    for reference in ('A','B'):
        comparisons['C400_vs_'+reference]=dict(reference=reference,candidate='C400',
            field_convention='Reusable pairing fields *_A denote reference; *_B denote C400',
            aggregate=paired(records[reference],records['C400'],classes),
            per_class=paired_classes(records[reference],records['C400'],classes))
    project=dict(scope='Fixed81FrameMiniValDevelopment_ProjectGT3559_NotOfficialMAP',aggregate=agg,scenes=scene,
        comparisons=comparisons,vertical_anomalies={n:anomalies(rs,classes) for n,rs in records.items()})
    write('reports/m8j10_project_comparison.json',project)
    official={'A':read('reports/m8j9_sources/resume/eval2/metrics_summary.json')}
    official.update({n:read(source/'official'/n/'metrics_summary.json') for n in ('B','C200','C400')})
    scalar=lambda m:dict(mAP=m['mean_ap'],NDS=m['nd_score'],**m['tp_errors'])
    official_rows={n:scalar(m) for n,m in official.items()}
    audit=read('reports/m8j9_GT_filter_audit.json')
    official_report=dict(name='LocalOfficialDevkitMiniVal_SpecifiedFrozenOutputPolicy_DevelopmentOnly',models=official_rows,
        per_class={n:m['mean_dist_aps'] for n,m in official.items()},
        per_class_AP_by_distance={n:m['label_aps'] for n,m in official.items()},
        per_class_TP_errors={n:{c:{k:None if np.isnan(v) else v for k,v in errors.items()} for c,errors in m['label_tp_errors'].items()} for n,m in official.items()},
        GT_records=3393,GT_support=audit['official_GT_filtered']['classes'],GT_unique_instances=audit['GT_instances_by_class'],
        official_nonapplicable_NaNs='RawOfficialFilesPreserved; aggregate copy uses null for NA only',
        deltas={ref:{k:official_rows['C400'][k]-official_rows[ref][k] for k in official_rows[ref]} for ref in ('A','B')})
    write('reports/m8j10_official_comparison.json',official_report)
    curves=dict(steps=[dict(step=r['global_step'],frame_index=r['frame_index'],scene=r['scene_name'],total_loss=r['total_loss'],
        classification_loss=r['classification_loss'],regression_loss=r['regression_loss'],gradient_norm_before_clip=r['gradient_norm_before_clip'],
        gradient_norm_after_clip=r['gradient_norm_after_clip'],parameter_delta_l2=r['parameter_delta_l2'],lr=r['actual_lr'],elapsed_s=r['elapsed_s']) for r in steps],
        training_loss_scope='DifferentSampleAtEachStep_NotFixedAggregateLearningCurve',
        eval_losses={str(s):{k:v for k,v in read(source/'train'/f'eval_{s:03}.json').items() if k in ('mean_total_loss','mean_losses','step')} for s in (200,400)},
        step0_loss='NotRecomputed_NoNewAForward; official and project A baselines reused',
        resources=dict(peak_step_allocated_bytes=max(s['peak_allocated_bytes'] for s in steps),peak_step_reserved_bytes=max(s['peak_reserved_bytes'] for s in steps),
            elapsed_per_step=distribution([s['elapsed_s'] for s in steps]),train_process_s=train['elapsed_s'],
            memory_scope='PyTorch allocator per-step; frozen Backbone; excludes driver/other processes'),
        freeze=dict(parameter_and_buffer_max_change=0.,checked_state_entries=337,canonical_trainable_parameters=141,
            exact_mapping_and_optimizer_unique=True,gradient_driven_updates_all_steps=all(s['gradient_driven_changed_parameters']>0 for s in steps)),
        final_reload=verify)
    write('reports/m8j10_learning_and_freeze.json',curves)
    figs=Path('reports/m8j10_figures');figs.mkdir(exist_ok=False)
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for key,label in [('total_loss','total'),('classification_loss','classification'),('regression_loss','regression')]:
        ax[0].plot([s['step'] for s in curves['steps']],[s[key] for s in curves['steps']],alpha=.6,label=label)
    ax[0].set_title('Training: changing samples, not fixed eval');ax[0].set_xlabel('optimizer step');ax[0].legend()
    names=list(official_rows);x=np.arange(len(names))
    ax[1].bar(x-.18,[official_rows[n]['mAP'] for n in names],.36,label='mAP');ax[1].bar(x+.18,[official_rows[n]['NDS'] for n in names],.36,label='NDS')
    ax[1].set_xticks(x);ax[1].set_xticklabels(names);ax[1].set_title('Official mini_val / fixed output policy');ax[1].legend()
    for n in names:ax[2].plot([r['recall'] for r in agg[n]['PR']],[r['precision'] for r in agg[n]['PR']],marker='.',label=n)
    ax[2].set_xlabel('Recall');ax[2].set_ylabel('Precision');ax[2].set_title('Project PR, 2m, cached candidates');ax[2].legend()
    fig.tight_layout();fig.savefig(figs/'comparison.png',dpi=150);plt.close(fig)
    # No per-object data are written to public reports.
    write(rep/'result.json',dict(status='OfflineComparisonCompleted',models=list(records),optimizer_steps=0,model_forward_calls=0,
        historical_AB_project_metrics_exact=True,final_checkpoint='C400',best_checkpoint_selection=False,
        created_at_utc=datetime.now(timezone.utc).isoformat(),code_commit=os.environ['M8J10_COMMIT']))
    print(json.dumps(official_rows,indent=2))


if __name__=='__main__':main()
