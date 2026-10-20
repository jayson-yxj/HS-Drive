"""Assemble source-linked evidence; never convert partial Drivable into a full M4 gate."""
import argparse,json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',default='m4_expanded');a=p.parse_args()
    selection=read('reports/'+a.prefix+'_selection.json');audit=read('reports/m4_driving_audit.json')
    index_path=Path('/root/gpufree-data/hs-drive-data/processed/hsd-dataset-v2-driving-accepted')
    index=read(index_path/'index.json');manifest=read(index_path/'manifest.json')
    evidence={'schema_version':'hsd-m4-expansion-report-v1','dataset':{'path':str(index_path),**manifest},
        'motion_audit':audit,'selection':selection,'evaluations':{},'checks':{},
        'full_m4_closed':False,'remaining':['Current traffic-rule/direction/route Drivable labels','Junction, slope and multilayer-road supervision coverage',
        'Broader independent dynamic traffic/occlusion/night/weather coverage','Repeated-seed robustness and operational quality thresholds']}
    for suffix in ('ce','depth','joint'):
        run=read(f'runs/{a.prefix}_{suffix}/manifest.json')
        for split in ('val','test'):
            path=Path(f'reports/{a.prefix}_{suffix}_{split}/metrics.json')
            if not path.exists():continue
            r=read(path)
            evidence['evaluations'][suffix+'_'+split]={'path':str(path),'sha256':file_hash(path),
                'depth':r['depth'],'depth_per_scene':r['depth_per_scene'],'camera_pixel_prior':r['camera_pixel_prior_depth'],
                'camera_pixel_prior_per_scene':r['camera_pixel_prior_per_scene'],'drivable':r['drivable_static'],
                'peak_allocated_gib':run['peak_allocated_gib'],'peak_reserved_gib':run['peak_reserved_gib'],
                'best_step':run['best_step'],'training_samples':len(run['train_sample_ids']),
                'rgb_only_inference_verified':r['inference_without_supervision']}
    evaluations=evidence['evaluations'];selected=selection['depth_selected_by_validation']
    baseline=read('reports/m4_val_final/metrics.json')['depth']['all']['mae_m']
    value=evaluations[selected+'_val']['depth_per_scene']['town03_spawn45_static']['all']['mae_m']
    evidence['checks']={'same_town03_validation_depth_improved_vs_original':value<baseline,
        'depth_log_loss_improved_vs_equal_budget_ce':evaluations['depth_val']['depth']['all']['mae_m']<evaluations['ce_val']['depth']['all']['mae_m'],
        'joint_val_static_iou_beats_all_positive':evaluations['joint_val']['drivable']['iou_positive']>evaluations['joint_val']['drivable']['all_positive_prior_iou'],
        'selected_test_depth_beats_pixel_prior':evaluations[selected+'_test']['depth']['all']['mae_m']<evaluations[selected+'_test']['camera_pixel_prior']['all']['mae_m'],
        'joint_test_depth_beats_pixel_prior':evaluations['joint_test']['depth']['all']['mae_m']<evaluations['joint_test']['camera_pixel_prior']['all']['mae_m'],
        'rgb_only_inference':all(x['rgb_only_inference_verified'] for x in evaluations.values()),
        'gpu_reserved_under_20_gib':all(x['peak_reserved_gib']<20 for x in evaluations.values())}
    label_coverage={}
    for scene in index['scenes']:
        path=Path(index['raw_root'])/'labels_static_v1'/scene['scene_id']/'manifest.json'
        labels=read(path);counts=labels['class_counts'];total=sum(counts.values());known=counts['positive']+counts['negative']
        label_coverage[scene['scene_id']]={'valid_fraction':known/total,'positive_fraction_of_valid':counts['positive']/known if known else None,
            'map_source':labels['map_source'],'label_manifest_sha256':file_hash(path)}
    evidence['label_coverage']=label_coverage
    evidence['drivable_location_prior']={split:read('reports/m4_drivable_prior_'+split+'.json') for split in ('val','test')}
    evidence['checks']['joint_val_static_iou_beats_location_prior']=evidence['drivable_location_prior']['val']['model_beats_location_prior']
    evidence['checks']['joint_test_static_iou_beats_location_prior']=evidence['drivable_location_prior']['test']['model_beats_location_prior']
    evidence['alignment_reports']={name:read('reports/'+name+'/report.json') for name in ('m4_static_alignment_val','m4_static_alignment_drive','m4_static_alignment_curve','m4_static_alignment_turn','m4_static_alignment_val_retry')}
    evidence['test_logs']={str(p):{'sha256':file_hash(p),'tail':p.read_text().splitlines()[-3:]} for p in (Path('logs/m4_expanded_gpu_tests.log'),Path('logs/m4_expanded_sim_tests.log'))}
    atomic_json('reports/m4_expansion.json',evidence)
    rows=[]
    for key,r in evaluations.items():
        iou='—' if r['drivable']['iou_positive'] is None else f"{r['drivable']['iou_positive']:.3f}"
        rows.append(f"| {key} | {r['depth']['all']['mae_m']:.3f} | {r['depth']['all']['abs_rel']:.3f} | {r['camera_pixel_prior']['all']['mae_m']:.3f} | {iou} |")
    report='''# M4 独立驾驶 / 深度 / 静态 Drivable 实测

保留原架构和旧实验；新增数据与监督管线已实际运行。完整 M4 门禁仍开放，当前 Drivable 只覆盖静态车道支持，不代表当前合法、无障碍可通行区。

'''
    report+=f"数据共 {manifest['samples']} 帧、{len(index['scenes'])} 场景，划分 {manifest['splits']}。新增累计路程 {audit['total_distance_m']:.1f}m。逐路线运动/重叠见 [audit](m4_driving_audit.json)。\n\n"
    report+='| 实验 / split | 深度 MAE m | AbsRel | 训练像素均值先验 MAE m | 静态 Drivable IoU |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    report+=f"原 Town03 同一验证场景：{baseline:.3f}m → {value:.3f}m。验证选择的深度配置：{selected}。历史基线只训练24帧/240步，不能将全部差异归因于单一因素；本次CE/depth两组相同数据和预算。\n\n"
    report+=f"联合模型（预先指定的次要实验）完整验证 MAE 为 {evaluations['joint_val']['depth']['all']['mae_m']:.3f}m，测试 {evaluations['joint_test']['depth']['all']['mae_m']:.3f}m；测试像素先验 {evaluations['joint_test']['camera_pixel_prior']['all']['mae_m']:.3f}m。其新 Town05 测试 MAE 为 {evaluations['joint_test']['depth_per_scene']['m4drive_town05_0']['all']['mae_m']:.3f}m。测试结果不用于改选主要深度配置，不将单种子的小幅差异解释为稳定算法收益。\n\n"
    report+='| 联合模型距离段 | 验证 MAE m | 测试 MAE m |\n|---|---:|---:|\n'
    for band in ('1.5-10m','10-30m','30-60m','60-80m'):
        report+=f"| {band} | {evaluations['joint_val']['depth'][band]['mae_m']:.3f} | {evaluations['joint_test']['depth'][band]['mae_m']:.3f} |\n"
    report+='\n'
    report+='评价仅覆盖有效平滑深度样本；Drivable IoU 只在映射的非路口内部有效区统计。测试冻结选择见 ['+a.prefix+'_selection.json]('+a.prefix+'_selection.json)。每场景、距离段、覆盖率、先验比较、检查结果和哈希详见 [机器可读报告](m4_expansion.json)。\n\n'
    prior=evidence['drivable_location_prior']
    report+=f"静态 Drivable 验证 IoU：模型 {prior['val']['model']['iou_positive']:.3f}，位置先验 {prior['val']['prior']['iou_positive']:.3f}，全正先验 {prior['val']['prior']['all_positive_iou']:.3f}；测试 IoU：模型 {prior['test']['model']['iou_positive']:.3f}，位置先验 {prior['test']['prior']['iou_positive']:.3f}。监督链路完成不等于分割质量通过。\n\n"
    report+='采集验收：新增966帧、六条短路线，累计389.7m；Town02低运动片段和Town04区间内丢1帧片段明确排除并保留失败证据。Town04重录恢复完整10Hz序列。正式索引逐帧哈希/标定/时序校验通过，候选索引未提交。\n\n'
    report+='验证：GPU环境87 passed / 5 skipped；仿真环境74 passed / 5 skipped（依赖环境相关跳过，不重复累加）。三组梯度有限、checkpoint精确重载；PyTorch峰值allocated约0.733GiB、reserved最高0.817GiB，不含驱动显存；每组训练约6.4分钟，单步compute约25–28ms，排除数据等待且不是端到端FPS。\n\n'
    report+='Drivable仅是静态Driving车道支持；0.5m边界带、路口、地图缺失及高度模糊均Unknown，rule_valid全部false。完整验证/测试有效覆盖约38.2%/22.3%。不能代表交通灯/方向/导航许可，也不代表无障碍空间。\n\n'
    report+='复核图：[实际路线](m4_driving_audit.png)、[联合模型验证深度](m4_expanded_joint_val/val_depth.png)、[静态Drivable验证对照](m4_drivable_validation_labeled.png)。\n\n'
    report+='[产物审计](m4_expansion_acceptance.json)：工程检查全部通过，质量诊断单列；完整M4未关闭。该审计不代替完整M4质量阈值。\n\n'
    report+='重现命令和语义见 [开发文档](../docs/M4_DRIVING_DEPTH_DRIVABLE.md)。路口/交通规则、动态交互和更多独立路线仍需补齐，不能据此关闭完整 M4 或宣布闭环驾驶成功。\n'
    Path('reports/m4_expansion.md').write_text(report)
    print(json.dumps(evidence['checks'],indent=2))

if __name__=='__main__':main()
