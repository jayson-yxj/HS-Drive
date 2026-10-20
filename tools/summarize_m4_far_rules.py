"""Summarize completed far-depth, road-context and signal-supervision experiments."""
import argparse,json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash
from tools.provenance import source_hashes


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m4_far_rules.json');a=p.parse_args()
    selection=read('reports/m4_road_selection.json');chosen=selection['selected']
    names=['m4_expanded_joint','m4_far_balanced','m4_road_point','m4_road_context'];evals={};runs={}
    for name in names:
        path=Path('reports')/(name+'_val')/'metrics.json';r=read(path);evals[name]=r
        runs[name]=read(Path('runs')/name/'manifest.json')
    legacy=read(f'reports/m4_road_{chosen}_legacy_val.json');test=read(f'reports/m4_road_{chosen}_test/metrics.json')
    signals={town:read('reports/m4signal_v2_'+town+'_audit/report.json') for town in ('town01','town04','town02')}
    priors={v:read('reports/m4_road_'+v+'_prior_val.json') for v in ('point','context')}
    prior_test=read('reports/m4_road_'+chosen+'_prior_test.json')
    point_config=dict(runs['m4_road_point']['config']);context_config=dict(runs['m4_road_context']['config'])
    point_config.pop('road_context');context_config.pop('road_context')
    actual_order={v:[(r['step'],r['sample_id']) for r in map(json.loads,Path('runs','m4_road_'+v,'metrics.jsonl').read_text().splitlines())] for v in ('point','context')}
    paired_sources=('models/depth_bev.py','training/runner.py','training/losses.py','datasets/training.py','tools/train_bev.py')
    checks={'all_new_runs_complete':all(runs[n]['status']=='complete' and runs[n]['last_step']==1200 for n in names[1:]),
        'finite_gradients_and_reload':all(runs[n]['all_gradients_finite'] and runs[n]['checkpoint_reload_exact'] for n in names[1:]),
        'same_train_plan':runs['m4_road_point']['train_sample_ids']==runs['m4_road_context']['train_sample_ids'],
        'same_actual_1200_step_order':len(actual_order['point'])==1200 and actual_order['point']==actual_order['context'],
        'same_training_implementation':all(runs['m4_road_point']['source_sha256'][p]==runs['m4_road_context']['source_sha256'][p] for p in paired_sources),
        'paired_config_differs_only_in_head':point_config==context_config,
        'same_road_labels':runs['m4_road_point']['road_overlay_manifest_sha256']==runs['m4_road_context']['road_overlay_manifest_sha256'],
        'evaluation_checkpoint_identity':all(evals[n]['checkpoint_sha256']==runs[n]['best_sha256']==file_hash(Path('runs')/n/'best.pt') for n in names),
        'selection_identity':all(selection['experiments'][v]['sha256']==file_hash(selection['experiments'][v]['val_metrics']) and
            selection['experiments'][v]['checkpoint_sha256']==runs['m4_road_'+v]['best_sha256'] for v in ('point','context')),
        'chosen_legacy_and_test_identity':legacy['checkpoint_sha256']==test['checkpoint_sha256']==runs['m4_road_'+chosen]['best_sha256'],
        'overlay_query_audit':all(read('reports/m4_road_overlay_exact_audit.json')['checks'].values()),
        'overlay_input_isolation':read('reports/m4_road_overlay_leakage.json')['passed'],
        'rgb_only':all(r['inference_without_supervision'] for r in evals.values()) and test['inference_without_supervision'],
        'signal_recording_audits':all(r['passed'] for r in signals.values()),
        'signal_strict_schema':read('reports/m4_signal_strict_schema.json')['passed'],
        'signal_loader_no_leakage':read('reports/m4_signal_loader_audit.json')['passed']}
    far=evals['m4_far_balanced'];old=evals['m4_expanded_joint']
    quality={'far_60_80_mae_improved':far['depth']['60-80m']['mae_m']<old['depth']['60-80m']['mae_m'],
        'near_mae_regressed':far['depth']['1.5-10m']['mae_m']>old['depth']['1.5-10m']['mae_m'],
        'selected_road_v2_beats_position_prior_val':priors[chosen]['model_beats_location_prior'],
        'selected_road_v2_beats_position_prior_test':prior_test['model_beats_location_prior'],
        'selected_road_v2_beats_all_positive_val':evals['m4_road_'+chosen]['drivable_static']['iou_positive']>priors[chosen]['prior']['all_positive_iou'],
        'selected_road_v2_beats_all_positive_test':test['drivable_static']['iou_positive']>prior_test['prior']['all_positive_iou'],
        'selected_road_v2_beats_each_scene_prior_val':all(s['iou_positive']>priors[chosen]['prior_per_scene'][sid]['iou_positive'] for sid,s in evals['m4_road_'+chosen]['drivable_static']['per_scene'].items()),
        'legacy_mask_iou_improved':legacy['drivable_static']['iou_positive']>old['drivable_static']['iou_positive']}
    report={'schema_version':'hsd-m4-far-rules-v1','engineering_passed':all(checks.values()),'checks':checks,'quality':quality,'source_sha256':source_hashes(),
        'selection':selection,'validation':{k:{'depth':v['depth'],'depth_per_scene':v['depth_per_scene'],'road':v['drivable_static']} for k,v in evals.items()},
        'selected_test':{'depth':test['depth'],'road':test['drivable_static']},'selected_legacy_mask_val':legacy['drivable_static'],
        'road_position_priors':priors,'road_position_prior_test':prior_test,'signals':signals,
        'gpu':{n:{k:runs[n][k] for k in ('peak_allocated_gib','peak_reserved_gib','wall_s','best_step')} for n in names[1:]},
        'full_m4_closed':False,'road_schema':'v2 junction Driving support, not full traffic permission',
        'remaining':['Far depth still inaccurate; near/far tradeoff unresolved','Drivable unseen environments and repeated seeds','Signal visibility/occlusion and image ROI supervision absent; rule validity is not visual visibility','Signal perception network not trained','Yellow decisions/route permission/yield/stop-sign compliance absent','Road direction supervision generated but no dedicated learned direction head'],
        'evidence':{str(p):file_hash(p) for p in [Path('reports/m4_road_selection.json'),Path('reports/m4_road_overlay_exact_audit.json'),Path('reports/m4_signal_loader_audit.json'),Path('logs/m4_rules_gpu_tests.log'),Path('logs/m4_rules_sim_tests.log')]}}
    atomic_json(a.output,report)
    lines=['# M4 远距离深度 / 跨地图 Drivable / 路口规则监督实测','',
        '沿用原模型主体、六摄输入与标签隔离合同。完整 M4 质量门禁保持开放；本轮实现与实测如下。','',
        '## 深度：有效改善远场，但不能整体替换基线','',
        '| 验证距离段 | 上轮 joint MAE m | 距离均衡 MAE m |','|---|---:|---:|']
    for b in ('all','1.5-10m','10-30m','30-60m','60-80m'):
        lines.append(f"| {b} | {old['depth'][b]['mae_m']:.3f} | {far['depth'][b]['mae_m']:.3f} |")
    lines+=['','四段CE/log-depth等权、按验证macro AbsRel选步。数据/种子/步数固定，但损失及选步准则同时变化。远场收益伴随近场退化，保留实验配置，不覆盖上一轮基线。',
        '',f"配对道路实验选定的{chosen}模型：全距离MAE={evals['m4_road_'+chosen]['depth']['all']['mae_m']:.3f}m，近场MAE={evals['m4_road_'+chosen]['depth']['1.5-10m']['mae_m']:.3f}m，60–80m MAE={evals['m4_road_'+chosen]['depth']['60-80m']['mae_m']:.3f}m。逐场景/分段完整结果保存在机器报告。",'',
        '## 道路：相同v2标签下比较预测头','',
        '| 预测头 | 全验证 IoU | 场景平均 IoU | 位置先验 IoU | 全正先验 IoU |','|---|---:|---:|---:|---:|']
    for v in ('point','context'):
        r=evals['m4_road_'+v]['drivable_static'];prior=priors[v]['prior']
        lines.append(f"| {v} | {r['iou_positive']:.3f} | {selection['experiments'][v]['macro_scene_iou']:.3f} | {prior['iou_positive']:.3f} | {prior['all_positive_iou']:.3f} |")
    lines+=['','| 验证场景 | 1×1头 IoU | 上下文头 IoU | 位置先验 IoU |','|---|---:|---:|---:|']
    for sid in evals['m4_road_point']['drivable_static']['per_scene']:
        values=[evals['m4_road_'+v]['drivable_static']['per_scene'][sid]['iou_positive'] for v in ('point','context')]
        lines.append(f"| {sid} | {values[0]:.3f} | {values[1]:.3f} | {priors[chosen]['prior_per_scene'][sid]['iou_positive']:.3f} |")
    lines+=['',f"选定头路口正样本召回率={evals['m4_road_'+chosen]['drivable_static']['junction']['recall_positive']:.3f}；该子集只有道路正样本，不能用它单独证明分类质量。"]
    lines+=['',f"验证选择 `{chosen}`；冻结选择后历史回归测试 IoU={test['drivable_static']['iou_positive']:.3f}，测试位置先验={prior_test['prior']['iou_positive']:.3f}，全正先验={prior_test['prior']['all_positive_iou']:.3f}。虽优于测试位置先验，仍低于全正先验；验证也未胜过先验，质量门禁未通过。Town05/Town10已在上轮观察过，不称新的盲测。",
        '',f"使用完全相同的旧v1有效mask重评：上一轮IoU={old['drivable_static']['iou_positive']:.3f}，本轮选定模型={legacy['drivable_static']['iou_positive']:.3f}。v1/v2标签覆盖不同，禁止直接比较不同mask的IoU。",
        '', 'v2新增连接车道条带形成的路口支持和有向车道监督；格子中心在条带内部才标正。地图外、边界、多层高度歧义保留Unknown；方向冲突仅屏蔽方向，保留道路正格。不能把路口道路几何解释为路权。原始/候选overlay保留，训练用road-overlay-v2-exact。',
        '', '## 信号监督：实际采集并加载，尚未训练视觉信号识别','',
        '| 场景 | 六摄帧 | 有效红灯 | 有效绿灯 | 无效/黄灯/切换 | 完整性 |','|---|---:|---:|---:|---:|---|']
    for town,r in signals.items():
        c=r['counts'];lines.append(f"| {town} | {r['integrity']['frames_verified']} | {c['valid_Red']} | {c['valid_Green']} | {c['invalid']} | {r['passed']} |")
    lines+=['','123帧、3个静止诊断场景组成独立信号index，按Town01/train、Town04/val、Town02/test隔离，不随机拆分同一灯态序列。Town02曾用于道路训练，不能作为整套系统的全新地图泛化证据。',
        '', '同一独占tick读取灯态，以前后frame校验绑定到相机帧；切换后2tick屏蔽。停止位置来自CARLA trigger-derived waypoint，不是已验收的精确喷漆停止线。稳定红灯=有停车义务，绿灯=仅该信号不要求停车，黄灯/关联不明无效；full_traffic_permission_valid永远false。',
        '', '规则有效掩码不代表信号在相机中可见；逐灯ROI、可见性/遮挡标注以及视觉识别网络仍待实现。',
        '', '首次采集0帧时遇到NumPy bool JSON类型错误，已修复并保留失败目录；Town07/Town06缺失，失败日志保留，按实际地图改用Town02。freeze()在0.9.15影响全场，已保存/恢复全场状态。',
        '', '## 工程验收与复现','',f"产物检查全部通过：{all(checks.values())}。GPU测试：{Path('logs/m4_rules_gpu_tests.log').read_text().splitlines()[-1]}；仿真测试：{Path('logs/m4_rules_sim_tests.log').read_text().splitlines()[-1]}。",
        '', '[实现与可执行命令](../docs/M4_FAR_ROAD_RULES.md) · [机器报告](m4_far_rules.json) · [道路几何抽查](m4_road_overlay_exact_audit.json) · [路口与方向可视化](m4_road_supervision_preview/road_supervision.png) · [灯态可视化](m4signal_v2_town01_audit/signal_timeline.png)',
        '', '下一步仍需解决近远距离损失权衡、更多独立地图/道路宽度与动态交互、视觉信号识别、路口让行/优先权/导航许可。当前无学习控制闭环通过声明。']
    Path(a.output).with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'engineering':checks,'quality':quality},indent=2))
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
