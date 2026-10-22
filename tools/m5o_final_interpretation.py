"""Finalize measured conclusions after sealed evaluation; no model or threshold mutation."""
import json
from pathlib import Path
from simulator.recorder import file_hash
from tools.m5o_common import write,asset,provenance
from datasets.m5o_benchmark import support


def update(path,fn,audit):
    p=Path(path);before=file_hash(p);d=json.loads(p.read_text());fn(d);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');audit.append(dict(path=path,before=before,after=file_hash(p)))


def main():
    audit=[];dataset=json.loads(Path('reports/m5o_dataset_support.json').read_text());test=json.loads(Path('reports/m5o_eval_sealed.json').read_text());cq=json.loads(Path('reports/m5o_candidate_quality_sealed.json').read_text());accept=json.loads(Path('reports/m5o_acceptance.json').read_text())
    assert all(not r['passed'] for r in test['curves']);assert accept['dataset_support']=='Passed_research_support_only'
    def acceptance(d):
        d.update(visual_localization_gain='NoGo',independent_risk_calibration='NoGo_selective_quality',generalization='NoGo',research_training='Passed_fit_only',safety_closed_loop='NotRun_OracleUnchanged_VisualProhibited',engineering_interfaces='Passed',overall='Partial',offline_object_center_pilot='NotApproved',reason='All planned overall location/candidate support targets met. Independent Calibration has no qualifying threshold in either seed; sealed Direct worse than GeometryPrior on identical targets and fails P95/severe-error gates. Stop current frozen-Mobile DirectCenter route without automatic step/layer/loss expansion.',delivery_tests=asset('runs/m5o_logs/full_tests_delivery.log'))
    update('reports/m5o_acceptance.json',acceptance,audit)
    def dataset_update(d):
        d['sealed_support']=accept['sealed_support'];d['sealed_capture']=asset('reports/m5o_capture_checks_sealed.json');d['all_planned_overall_support_targets_met']=True;d['class_camera_generalization_passed']=False;d['actual_geographic_isolation']=asset('reports/m5o_actual_split_geometry.json');d['lineage_audit']=asset('reports/m5o_leakage_audit_final.json')
    update('reports/m5o_dataset_support.json',dataset_update,audit)
    def benchmark_update(d):
        d['paired_method_analysis']=asset('reports/m5o_paired_method_analysis.json');d['inference_equivalence']=asset('reports/m5o_inference_equivalence.json');d['quality_status']='NoGo';d['engineering_status']='Completed'
    update('reports/m5o_localization_benchmarks.json',benchmark_update,audit)
    def scaling_update(d):
        d['conclusion']='Independent locations help Direct mean error (4.95->3.61m, 4.31->3.66m) and severe rate, but neither seed beats geometry prior. Seed102 P95 is not monotonic (13.88->12.28->12.47m); no budget extension. Fixed budget does not prove universal feature impossibility.'
        d['paired_location_analysis']=asset('reports/m5o_paired_method_analysis.json')
    update('reports/m5o_data_scaling.json',scaling_update,audit)
    def risk_update(d):
        d['quality_status']='NoGo';d['selected_thresholds']={'101':1.1,'102':1.1};d['selected_coverage']=0;d['fixed_error_rate_curves_are_posthoc_not_threshold_selection']=True;d['candidate_burden']={s:asset(f'reports/m5o_all_candidate_burden_{s}.json') for s in ['train','val','calibration','sealed']}
    update('reports/m5o_risk_coverage.json',risk_update,audit)
    def candidate_update(d):
        d['full_ledger']={s:asset(f'reports/m5o_all_candidate_burden_{s}.json') for s in ['train','val','calibration','sealed']};d['sealed_failure_partition']=[dict(seed=101,visible=363,missed=56,ambiguous_matched=30,unique_direct_over5m=98,unique_direct_within5m=179),dict(seed=102,visible=363,missed=52,ambiguous_matched=24,unique_direct_over5m=94,unique_direct_within5m=193)]
    update('reports/m5o_candidate_quality.json',candidate_update,audit)
    p=Path('reports/m5o_summary.md');s=p.read_text().replace('NoGo_or_unresolved','NoGo').replace('Sealed计划6','Sealed6（实际全部完成）')
    s=s.replace('## 2–3. 数据规模影响与Direct相对Geometry Prior','''封存失败分解（互不重叠）：seed101=漏检56 + 关联歧义30 + 唯一关联但Direct>5m 98 + 唯一关联且≤5m 179 =363；seed102=52+24+94+193=363。因此候选/关联造成约23.7%/20.9%的目标不能进入可靠定位，定位器本身还产生大量严重错误。未匹配候选3963/2975不在这些已知目标分母中，必须另行保留为未知负担。

## 2–3. 数据规模影响与Direct相对Geometry Prior''')
    s=s.replace('真实RTX4090训练共12臂','''从5个训练位置扩到20个，Direct平均XY误差由4.95→3.61m、4.31→3.66m；P95由14.88→9.98m、13.88→12.47m。第二种子10→20位置的P95略升，尾部改善不稳定。两种子在相同可定位目标上均逊于Geometry Prior。封存同目标、位置宏平均差为Direct比Prior多1.45m/1.33m；描述性位置bootstrap区间均为正，不能当成跨地图总体显著性证明。

Direct训练Loss降至约0.001，Validation性能仍差；数据支持不足是原瓶颈之一，扩大后仍不能解释掉特征/回归映射的泛化问题。固定600步试验不能证明任何预算下都不可能改善，但不支持继续当前路线自动增加训练。

真实RTX4090训练共12臂''')
    s=s.replace('最大峰值PyTorch分配 20.5MiB。','最大峰值PyTorch分配 20.5MiB（缓存特征上的小Head训练，不含完整Backbone；634帧训练特征缓存190.5s、峰值237.0MiB）。')
    s=s.replace('## 6–7. 架构决策与下一阶段','''封存分层Direct P95：Vehicle 12.47/10.71m，Person 6.99/7.37m，Cone 9.11/9.09m；0–10m 3.51/5.22m、10–20m 7.55/7.40m、20–40m 12.83/12.41m。没有类别证明可可靠定位。封存右前摄仅7/7、右后摄3/4个可靠候选且各只有1位置，不能解释其偶然较低误差为可靠。其他四摄P95大多仍8.6–12.4m。Calibration锥桶仅17/13，类别风险校准支持不足。

## 6–7. 架构决策与下一阶段''')
    s=s.replace('本轮不以Loss或平均误差替代尾部、严重错误、独立Calibration和全候选负担门禁。若新数据支持不足，不能声称充分数据下Head失败；若支持充分但固定方法仍不达标，停止自动加层/调Loss/延长预算。','本轮已达到预先计划的总体研究支持目标，但固定Direct路线仍未通过尾部、严重错误、独立Calibration与同目标基线门禁，因此停止当前冻结Mobile+DirectCenter Head的自动扩展，不追加层数、Loss或训练步数。总体支持达标不等于类别、全六摄或40–80m支持充分。')
    s=s.replace('测试：324 passed, 6 skipped in 18.20s','测试：324 passed, 6 skipped in 18.95s（交付复测；原18.20s日志也保留）')
    p.write_text(s)
    p=Path('docs/M5O_ARCHITECTURE_DECISION.md');s=p.read_text().replace('状态：等待预声明实验；最终以 `reports/m5o_acceptance.json` 和 `m5o_summary.md` 为准。','''状态：**停止当前冻结Mobile+DirectCenter Head路线；M5-O总体Partial；不批准Offline Object-Center Pilot。**

实际完成38位置、1223个六摄配置，Train/Val/Calibration/Sealed支持分别642–658 / 198–202 / 199–205 / 269–279可靠候选。两个种子的固定预算结果均未稳定优于同候选Geometry Prior；封存Direct P95 12.33/10.66m，严重错误35.4%/32.8%。独立Calibration无合格阈值，全部准入拒绝。新增数据改善均值，但未解决尾部；不再自动增加层数、Loss或步数。

工程合同与有限训练管线通过，定位泛化与选择性风险未通过，纯视觉安全没有验证。未来可复用此Benchmark预声明研究其他成熟3D表示，但本轮没有验证替代方案，不能直接批准位置型Pilot。Oracle Planner/仿真工程可独立推进。''')
    p.write_text(s)
    p=Path('docs/M5O_IMPLEMENTATION.md');s=p.read_text()+'''\n## 本轮实际完成\n\n38位置，1223个六摄配置；真实4090上12臂各600步（没有追加预算）。Train/Val/Calibration/Sealed有效唯一候选分别642/658、198/202、199/205、269/279。模型选择、风险校准和封存严格分组；封存无再次调参。最终Partial、当前Head NoGo、Pilot不批准。\n\n补充汇总命令：\n\n```bash\n.venv-gpu/bin/python -m tools.m5o_paired_analysis\n.venv-gpu/bin/python -m tools.m5o_final_interpretation\n```\n\n`final_interpretation`在summary/report首次生成后补入最终测量结论并记录报告前后SHA；不得重跑覆盖。源码Checkpoint选择/推理与预封存资产保持原字节；仅最终报告编排变化。完整实际顺序还包括Train侧车完成后的early候选审计、日志中的watcher恢复。初始通路失败和全部运行日志均保留。\n''';p.write_text(s)
    write('reports/m5o_final_interpretation.json',dict(kind='post_evaluation_report_only_no_selection',revisions=audit,provenance=provenance(),decisions=dict(current_head='Stop',pilot='NotApproved',occupancy='NotRun',visual_control='Prohibited')))

if __name__=='__main__':main()
