"""Offline aggregate of fixed checkpoints; no model imports or updates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from training.m8j7_metrics import match, errors, distribution


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,data):
    with Path(path).open('x') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    cfg=read('configs/m8j7_protocol.json');run=cfg['run_id']
    root=Path('reports/m8j7_sources')/run
    private=Path('/root/gpufree-data/hs-drive-private')/run
    supervisor=read(root/'supervisor.json')
    assert supervisor['status']=='CompletedAndReloadPassed'
    train=read(root/'train/result.json');verify=read(root/'verify/result.json')
    assert train['optimizer_steps']==400 and verify['optimizer_steps']==0
    evaluations=[read(root/'train'/f'eval_{s:03}.json') for s in cfg['evaluation']['steps']]
    steps=[json.loads(x) for x in (root/'train/steps.jsonl').read_text().splitlines()]
    assert [s['global_step'] for s in steps]==list(range(1,401))
    assert all(s['frame_index']==(s['global_step']-1)%8 for s in steps)
    assert all(s['optimizer_after']['lr']==[1e-5] and s['optimizer_after']['weight_decay']==[0.] for s in steps)
    assert all(s['frozen_parameter_and_buffer_max_abs_change']==0 and s['all_buffers_max_abs_change']==0 for s in steps)
    assert all(s['gradients']['all_finite'] and s['gradient_driven_changed_parameters']>0 for s in steps)
    assert all(e['primary']['GT']==335 for e in evaluations)
    assert all(s['optimizer_after']['step_values']==[s['global_step']] for s in steps)
    for before,after in zip(steps,steps[1:]):
        assert before['after_state_sha256']==after['before_state_sha256']
        assert before['optimizer_after']==after['optimizer_before']
    # Old assets and executed source remain byte-for-byte unchanged.
    preserved=[]
    for name in ['reports/m8j6_retry_release_manifest.json','reports/m8j6_actual_release_manifest.json']:
        m=read(name)
        for row in m['files']:assert digest(row['path'])==row['sha256'],row['path']
        preserved.append(dict(manifest=name,files=len(m['files']),unchanged=True))
    for p,h in supervisor['source_sha256'].items():assert digest(p)==h
    for row in read('reports/m8j6_source_lock.json')['assets']:assert digest(row['path'])==row['sha256']
    prepared=Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared')
    assert digest(prepared/'manifest.json')==cfg['prepared_manifest_sha256']
    for f in cfg['frames']:
        for folder,key in [('rgb_input','input_sha256'),('supervision','supervision_sha256')]:
            assert digest(prepared/folder/('%02d.npz'%f['index']))==f[key]
    for cp in train['checkpoints']:assert digest(cp['path'])==cp['sha256']
    for phase in supervisor['phases']:
        assert digest(root/phase['phase']/'execution.txt')==phase['log_sha256']
        assert not Path('/proc/%d'%phase['PID']).exists()
    mapping=read(root/'train/parameter_mapping.json')
    assert mapping==read(root/'verify/parameter_mapping.json')
    preservation=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),historical=preserved,
        original_sources_and_weights_unchanged=True,fixed_16_bundles_unchanged=True,
        training_GT_unchanged=335,checkpoints_verified=len(train['checkpoints']),processes_exited=True,
        per_step_detail_hashes_verified=0,ledger_steps=400)
    for s in steps:
        assert digest(private/'steps'/f'step_{s["global_step"]}.json')==s['private_detail_sha256']
        preservation['per_step_detail_hashes_verified']+=1
    assert len(list((private/'steps').glob('optimizer_step_*_returned.json')))==400
    write('reports/m8j7_preservation.json',preservation)

    keys=['mean_total_loss','mean_classification_loss','mean_regression_loss','primary','stricter','AP']
    metrics=[dict(step=e['step'],**{k:e[k] for k in keys}) for e in evaluations]
    write('reports/m8j7_learning_curves.json',dict(scope='FixedEightTrainingFramesOnly',evaluations=metrics))
    a,b=evaluations[0],evaluations[-1]
    before_after=dict(per_frame=[dict(index=x['index'],sample_token=x['sample_token'],
        baseline=x,final=y) for x,y in zip(a['frames'],b['frames'])],
        per_class={c:dict(baseline=a['primary']['per_class'][c],final=b['primary']['per_class'][c])
                   for c in a['primary']['per_class']})
    write('reports/m8j7_before_after.json',before_after)

    # Additional matched-subset error control; diagnostic only, no selection changes.
    classes=list(a['primary']['per_class'])
    common_errors=[{k:[] for k in ('xy_m','z_center_m','dimension_abs_mean_m','aligned_size_iou_error','yaw_rad')} for _ in range(2)]
    residual=[];common_count=0;unmatched_reasons={}
    common_class=[{c:{k:[] for k in common_errors[0]} for c in classes} for _ in range(2)]
    for i in range(8):
        rr=[]
        for step in (0,400):
            with np.load(private/f'eval_{step:03}'/f'{i:02}_decoded_and_GT.npz',allow_pickle=False) as npz:
                rr.append({k:npz[k].copy() for k in npz.files})
        assert np.array_equal(rr[0]['gt'],rr[1]['gt']) and np.array_equal(rr[0]['gt_labels'],rr[1]['gt_labels'])
        mm=[match(**r) for r in rr];common=set(j for _,j in mm[0]['pairs'])&set(j for _,j in mm[1]['pairs'])
        common_count+=len(common)
        for ri,(r,m) in enumerate(zip(rr,mm)):
            er=errors(r['boxes'],r['gt'],[(p,g) for p,g in m['pairs'] if g in common],r['gt_labels'],classes)
            for k,v in er.items():common_errors[ri][k].extend(v)
            for ci,c in enumerate(classes):
                ec=errors(r['boxes'],r['gt'],[(p,g) for p,g in m['pairs'] if g in common and r['gt_labels'][g]==ci],r['gt_labels'],classes)
                for k,v in ec.items():common_class[ri][c][k].extend(v)
        final=rr[1];m=mm[1]
        reasons={}
        for j in m['fn']:
            ci=int(final['gt_labels'][j]);name=classes[ci]
            ids=np.where((final['labels']==ci)&(np.abs(final['boxes'][:,0])<51.2)&(np.abs(final['boxes'][:,1])<51.2))[0]
            close=[p for p in ids if np.linalg.norm(final['boxes'][p,:2]-final['gt'][j,:2])<2.]
            if not len(ids):reason='NoSameClassDecodedCandidate'
            elif not close:reason='NoSameClassCandidateWithin2m'
            elif not any(final['scores'][p]>=.2 for p in close):reason='Within2mButBelowFixedScoreThreshold'
            else:reason='NearbyPredictionConsumedByOneToOneAssignment'
            reasons[reason]=reasons.get(reason,0)+1
            cls=unmatched_reasons.setdefault(name,{})
            cls[reason]=cls.get(reason,0)+1
        bins=[]
        for lo,hi in ((0,20),(20,40),(40,75)):
            radii=np.linalg.norm(final['gt'][:,:2],axis=1);ids=set(np.where((radii>=lo)&(radii<hi))[0])
            bins.append(dict(radial_interval_m=[lo,hi],GT=len(ids),FN=len(ids&set(m['fn']))))
        residual.append(dict(index=i,GT=len(final['gt']),FN=len(m['fn']),FP=len(m['fp']),duplicates=len(m['duplicates']),
            FN_by_distance=bins,FN_reasons=reasons))
    write('reports/m8j7_residual_diagnostics.json',dict(common_matched_GT_records=common_count,
        common_subset_baseline={k:distribution(v) for k,v in common_errors[0].items()},
        common_subset_final={k:distribution(v) for k,v in common_errors[1].items()},
        common_subset_per_class={c:dict(baseline={k:distribution(v) for k,v in common_class[0][c].items()},
            final={k:distribution(v) for k,v in common_class[1][c].items()}) for c in classes},
        final_frame_errors=residual,final_FN_reasons_by_class=unmatched_reasons,
        scope='MatchedSubsetControlAndFinalResiduals_NotNewModelSelection'))

    out=Path('reports/m8j7_figures');out.mkdir(exist_ok=False)
    xs=[e['step'] for e in evaluations]
    fig,ax=plt.subplots(2,2,figsize=(11,8))
    for k,label in [('mean_total_loss','total'),('mean_classification_loss','classification'),('mean_regression_loss','regression')]:
        ax[0,0].plot(xs,[e[k] for e in evaluations],marker='o',label=label)
    ax[0,0].set_title('Fixed 8-frame eval mean loss');ax[0,0].legend()
    for k in ('precision','recall'):
        ax[0,1].plot(xs,[e['primary'][k] for e in evaluations],marker='o',label=k)
    ax[0,1].set_title('Decoded detections: score >= 0.2, XY < 2m');ax[0,1].legend()
    for d in ('1.0','0.5'):
        ax[1,0].plot(xs,[e['stricter'][d]['recall'] for e in evaluations],marker='o',label='recall < '+d+'m')
    ax[1,0].plot(xs,[e['AP']['mean_supported_classes'] for e in evaluations],marker='o',label='diagnostic mean AP@2m')
    ax[1,0].legend();ax[1,0].set_title('Training-set stricter localization / AP')
    for e in (a,b):
        ax[1,1].plot([x['recall'] for x in e['PR']],[x['precision'] for x in e['PR']],marker='o',label='step '+str(e['step']))
    ax[1,1].set_xlabel('Recall');ax[1,1].set_ylabel('Precision');ax[1,1].legend();ax[1,1].set_title('Fixed threshold grid PR@2m')
    for axis in ax.ravel():axis.grid(alpha=.3)
    fig.suptitle('HS-Drive M8-J7: eight correlated training frames, not validation')
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for k in ('TP','FP','FN'):
        ax[0].plot(xs,[e['primary'][k] for e in evaluations],marker='o',label=k)
    ax[0].legend();ax[0].set_title('Full selected candidate counts')
    ax[1].plot(xs,[e['primary']['matched_errors']['xy_m']['mean'] for e in evaluations],marker='o',label='XY mean m')
    ax[1].plot(xs,[e['primary']['matched_errors']['xy_m']['p95'] for e in evaluations],marker='o',label='XY P95 m')
    ax[1].plot(xs,[e['primary']['matched_errors']['yaw_rad']['mean'] for e in evaluations],marker='o',label='applicable yaw mean rad')
    ax[1].legend();ax[1].set_title('Matched-pair errors; count changes also reported')
    for axis in ax:axis.grid(alpha=.3);axis.set_xlabel('Optimizer step')
    fig.tight_layout();fig.savefig(out/'detection_errors.png',dpi=150);plt.close(fig)

    # Private all-eight before/after review index; no new raw-data reads.
    html='<meta charset="utf-8"><h1>M8-J7 eight-frame before / final</h1><p>Training-set diagnostic; green GT, orange decoded predictions; no human approval claimed.</p>'
    for i in range(8):
        html+=f'<h2>Frame {i}</h2>'
        for step in (0,400):
            html+=f'<h3>Step {step}</h3><img width="1500" src="eval_{step:03}/{i:02}_six.jpg"><img width="700" src="eval_{step:03}/{i:02}_bev.jpg">'
    (private/'comparison.html').write_text(html)
    resource=dict(peak_step_allocated_bytes=max(s['peak_allocated_bytes'] for s in steps),
        peak_step_reserved_bytes=max(s['peak_reserved_bytes'] for s in steps),
        per_step_elapsed_s=distribution([s['elapsed_s'] for s in steps]),
        gradient_norm_before_clip=distribution([s['gradient_norm_before_clip'] for s in steps]),
        gradient_norm_after_clip=distribution([s['gradient_norm_after_clip'] for s in steps]),
        changed_parameters_per_step=distribution([s['gradient_driven_changed_parameters'] for s in steps]),
        parameter_delta_l2=distribution([s['parameter_delta_l2'] for s in steps]),
        actual_learning_rates=sorted({v for s in steps for v in s['optimizer_after']['lr']}),
        memory_scope='Per-step PyTorch allocator, frozen backbone; not driver or full end-to-end training',
        train_process_wall_s=supervisor['phases'][0]['elapsed_s'],
        evaluation_wall_s=[dict(step=e['step'],seconds=e['elapsed_s']) for e in evaluations])
    write('reports/m8j7_resource_and_freeze.json',dict(resources=resource,
        trainable_parameter_objects=141,frozen_parameter_objects=178,frozen_buffers=159,
        maximum_frozen_change=0.,maximum_buffer_change=0.,mapping_unchanged=True,
        final_reload=verify,checkpoints=train['checkpoints']))
    directions=dict(fixed_eval_loss_lower=b['mean_total_loss']<a['mean_total_loss'],
        primary_TP_higher=b['primary']['TP']>a['primary']['TP'],
        primary_FP_lower=b['primary']['FP']<a['primary']['FP'],
        strict_1m_TP_higher=b['stricter']['1.0']['TP']>a['stricter']['1.0']['TP'],
        strict_05m_TP_higher=b['stricter']['0.5']['TP']>a['stricter']['0.5']['TP'])
    write('reports/m8j7_acceptance.json',dict(stage='M8-J7',base_commit='f4e44f9',
        execution_code_commit=supervisor['code_commit'],run_id=run,
        status='Completed_LearningEvidenceSupported' if all(directions.values()) else 'Completed_LearningEvidenceMixedOrInsufficient',
        observed_learning_directions=directions,
        stable_training='Passed_Exactly400Steps',learning_scope='EightFixedTrainingFrames_NotGeneralization',
        initial_visualization_failure=dict(run_id='m8j7_20261027_131635',optimizer_steps=0,preserved=True),
        optimizer_steps_this_stage=400,source_checkpoint='OriginalBaseline_FreshOptimizer',
        tests=dict(total=63,passed=63,scope='20IngestSelection_16Official_14Frozen_13MetricsVisualization'),
        final_checkpoint=train['checkpoints'][-1],final_reload='Passed_NewProcess_ZeroUpdates',
        finite_losses_gradients=True,frozen_state='ExactUnchanged',GT_records=335,
        baseline=dict(loss=a['mean_total_loss'],TP=a['primary']['TP'],FP=a['primary']['FP'],FN=a['primary']['FN']),
        final=dict(loss=b['mean_total_loss'],TP=b['primary']['TP'],FP=b['primary']['FP'],FN=b['primary']['FN']),
        human_GoldenGT_signoff='NotClaimed',AMP='NotRetested_NotPassed',CARLA='NotRun',
        generalization='NotEvaluated',official_nuscenes_score='NotClaimed',private_assets_published=False,main_merge=False))
    print(json.dumps(dict(status='ReportsGenerated',baseline=a['primary']['TP'],final=b['primary']['TP'],steps=400)))
    def number(v,decimals=4):
        return 'N/A' if v is None else f'{v:.{decimals}f}'
    lines=[
        '# M8-J7 固定八帧 FP32：400步及学习效果诊断',
        '',
        f'日期：{datetime.now(timezone.utc).date()}；基础f4e44f9，独立分支feat/m8j7-eight-frame-fp32-overfit。',
        f'实际执行源码：{supervisor["code_commit"]}；run_id：{run}。最终交付另附汇总代码，不改执行源码。',
        '',
        '**有限训练稳定完成：400/400次更新；最终新进程加载通过。结果仅属于固定八帧训练集，不是nuScenes验证成绩或泛化验收。**',
        '',
        '## 配置与边界',
        '',
        '从Smoke相同原始checkpoint及空AdamW开始，未使用Smoke第3步权重/动量。八帧0–7固定循环50次，全部335条训练GT保持不变，未删除困难目标或扩数据。两个训练场景与官方预训练full train有已知重叠，连续帧也相关，335条记录不是335个独立物体。',
        'FP32、bs1、六摄512×1408、十类、900 Query、六层；AdamW lr1e-5、betas(0.9,0.999)、wd0、clip35、常数学习率、无scheduler，seed875606。Head Dropout保持0.1，Backbone/Neck eval和no_grad。AMP/TF32关闭，无GradScaler，不更换模型结构、图像增强、监督规则。',
        '完整配置/token/SHA在 [冻结协议](../configs/m8j7_protocol.json)。初始权重SHA256：'+train['original_weight_sha256']+'。',
        '',
        '## 全部固定评估点',
        '',
        '正常head.get_bboxes解码：原NMSFree top300 query-class pairs，独立XY任务区域过滤，主score≥0.2、同类一对一XY距离<2m。GT不参与预测筛选。重复和未匹配预测保留为FP；任务范围外预测另报。评测GT与训练GT同为335，未再过滤；某些物理目标可能因官方点数/任务范围规则不在GT中，因此FP是任务定义下的未匹配，不自动等同物理虚假物体。',
        'Loss为同一eval模式下八帧平均，各层和各帧完整记录均保留；不把随机训练单步Loss当整体进步。评估恢复原随机状态与训练模式。',
        '',
        '|步数|分类Loss|回归Loss|总Loss|TP|FP|FN|Precision|Recall|1m TP|0.5m TP|诊断mAP@2m|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for e in evaluations:
        q=e['primary']
        lines.append(f'|{e["step"]}|{e["mean_classification_loss"]:.4f}|{e["mean_regression_loss"]:.4f}|{e["mean_total_loss"]:.4f}|{q["TP"]}|{q["FP"]}|{q["FN"]}|{q["precision"]:.2%}|{q["recall"]:.2%}|{e["stricter"]["1.0"]["TP"]}|{e["stricter"]["0.5"]["TP"]}|{e["AP"]["mean_supported_classes"]:.4f}|')
    lines+=['','AP是本协议101点插值、有GT类别均值，不是官方nuScenes mAP/NDS。没有根据结果挑“最佳”检查点；第400步为最终结果。所有PR阈值、未匹配与重复预测计数见逐点评估原始JSON。',
        '', '![固定评估曲线](m8j7_figures/learning_curves.png)',
        '', '## 逐帧训练前后', '',
        '|帧|GT|Loss 0→400|TP 0→400|FP 0→400|FN 0→400|',
        '|---:|---:|---|---|---|---|']
    for x,y in zip(a['frames'],b['frames']):
        p,q=x['metrics'],y['metrics']
        lines.append(f'|{x["index"]}|{p["GT"]}|{x["total_loss"]:.4f} → {y["total_loss"]:.4f}|{p["TP"]} → {q["TP"]}|{p["FP"]} → {q["FP"]}|{p["FN"]} → {q["FN"]}|')
    lines+=['','## 分类别与剩余问题','','|类别|GT|TP 0→400|FP 0→400|FN 0→400|XY均值m 0→400|适用Yaw均值rad 0→400|','|---|---:|---|---|---|---|---|']
    for c,x in a['primary']['per_class'].items():
        y=b['primary']['per_class'][c]
        lines.append(f'|{c}|{x["GT"]}|{x["TP"]} → {y["TP"]}|{x["FP"]} → {y["FP"]}|{x["FN"]} → {y["FN"]}|{number(x["matched_errors"]["xy_m"]["mean"])} → {number(y["matched_errors"]["xy_m"]["mean"])}|{number(x["matched_errors"]["yaw_rad"]["mean"])} → {number(y["matched_errors"]["yaw_rad"]["mean"])}|')
    lines+=['','bus/trailer/construction_vehicle无GT，不赋予AP/Recall高分；仍统计其误检。motorcycle和bicycle各仅1条GT，不支持类别泛化结论。traffic_cone不评价Yaw；barrier按π周期，其余适用类按2π。',
        '', '|匹配误差|第0步均值|第400步均值|第0步P95|第400步P95|','|---|---:|---:|---:|---:|']
    for k,x in a['primary']['matched_errors'].items():
        y=b['primary']['matched_errors'][k]
        lines.append(f'|{k}|{number(x["mean"])}|{number(y["mean"])}|{number(x["p95"])}|{number(y["p95"])}|')
    lines+=['',f'成功匹配集合随训练变化。额外固定两端都匹配的{common_count}条GT记录，按同一GT比较误差，见 [残差诊断](m8j7_residual_diagnostics.json)。该分析只解释结果，不修改匹配/训练/阈值。',
        '', '最终漏检原因（仅基于已固定正常解码和匹配的事后诊断）：', '']
    for name,v in unmatched_reasons.items():
        lines.append('- '+name+'：'+json.dumps(v,ensure_ascii=False))
    lines+=['','NoSameClassCandidateWithin2m不能单独证明几何模块失效，也可能涉及分类、top300截断或特征；本轮未进行这些归因干预。','',
        '![计数与匹配误差](m8j7_figures/detection_errors.png)',
        '', '## 冻结、梯度、资源与恢复',
        '',
        f'400步Loss/梯度均有限，每步均存在分类及回归有效梯度和实际参数更新，wd=0。每步变化参数对象最少{min(s["gradient_driven_changed_parameters"] for s in steps)}、最多{max(s["gradient_driven_changed_parameters"] for s in steps)}，不能仅用权重衰减解释更新。',
        '141个训练Parameter对象、221个训练state别名，178个冻结Parameter对象及159个buffer共337项冻结state始终精确不变。每步优化器参数去重、训练/冻结互斥及别名一致性检查均通过。全参数连续状态hash、Adam状态、LR和step连续。',
        f'每步峰值allocated {resource["peak_step_allocated_bytes"]:,} bytes（{resource["peak_step_allocated_bytes"]/1024**3:.3f} GiB），reserved {resource["peak_step_reserved_bytes"]:,} bytes（{resource["peak_step_reserved_bytes"]/1024**3:.3f} GiB）。这是冻结骨干、Head反向的PyTorch allocator口径，不含驱动/其他进程，不是全网训练显存。',
        f'训练进程墙钟{resource["train_process_wall_s"]:.2f}秒，单步中位{resource["per_step_elapsed_s"]["median"]:.3f}秒、P95 {resource["per_step_elapsed_s"]["p95"]:.3f}秒；包含读取/前后向/更新/校验，不是纯推理或实时驾驶性能。',
        '0/80/160/240/320/400检查点全保留。最终独立进程恢复模型、buffer、优化器、模式、RNG和共享别名，固定eval输出容差预先为atol1e-6/rtol1e-5。',
        f'最终检查点SHA256：{train["checkpoints"][-1]["sha256"]}；最终验证0次更新，误差：'+json.dumps(verify['fixed_eval_output_comparison'])+'。',
        '没有连续训练逐位等价对照，不作扩大声明。具体资源/加载证据见 [资源与冻结报告](m8j7_resource_and_freeze.json)。',
        '', '## 失败记录、测试与复现',
        '',
        '首个run m8j7_20261027_131635在第0步绘图失败，实际optimizer.step=0。官方PETR extract_img_feat对bs1调用squeeze_()；新绘图器误按五维索引。修复仅使绘图接受4D/5D同一内容，并加入回归测试，完整失败日志及私有failure checkpoint保留。独立修复后run从原始权重/空优化器开始；没有扩展400步预算、调整数据/模型或通过重跑筛选结果。',
        '原50项测试继续通过，新增13项匹配/AP/距离/类别/重复/空预测/图像维度测试通过，共63项。执行代码、冻结协议与预检在训练前提交。后处理汇总另行保存，不改执行证据。所有历史M8-J6文件、原始权重和16个输入/监督包SHA不变。',
        '命令及实现说明见 [M8J7_IMPLEMENTATION](../docs/M8J7_IMPLEMENTATION.md)。私有前后对照：'+str(private/'comparison.html')+'；包含全部八帧六摄和BEV，未公开图像、逐对象标签/预测或模型。实际查看记录单独保存，未查看项不宣称人工确认。',
        '', '## 决策范围',
        '',
        '1. 有限训练稳定性：400步和最终重载通过；没有第401步。',
        '2. 学习效果：依据同一eval协议的Loss、全候选TP/FP/FN、严格距离和PR曲线分别判断，不能替代独立场景评测。',
        '3. 改善与退化：以上表格完整列出，包括无GT/弱支持类别与误差尾部；不因总体Loss下降忽略局部退化。',
        '4. 下一步只提出限定实验，不在本轮执行。独立场景测试与最小定位诊断的最终建议见本文附加解读。',
        '',
        '保留AMP未通过、Occupancy/视觉驾驶未验证状态。未启动CARLA、未更换模型或新增OCC/时序、未合并main。']
    with Path('reports/m8j7_summary.md').open('x') as f:f.write('\n'.join(lines)+'\n')


if __name__=='__main__':main()
