"""Summarize completed bounded experiment; no new evaluations or updates."""
import json
from pathlib import Path
from datetime import datetime,timezone
from datasets.m8j6_selection import sha256


def read(p):return json.loads(Path(p).read_text())
def main():
    cfg=read('configs/m8j10_protocol.json');base=Path('reports/m8j10_sources')/cfg['run_id']
    v=read('reports/m8j10_verification.json');o=read('reports/m8j10_official_comparison.json')
    p=read('reports/m8j10_project_comparison.json');l=read('reports/m8j10_learning_and_freeze.json')
    e=read('reports/m8j10_exposure.json');data=read(base/'prepare/result.json');context=read('reports/m8j10_data_context.json')
    scores=o['models'];c=scores['C400']
    changes={}
    for name in ('A','B'):
        a=scores[name]
        changes[name]=dict(mAP=c['mAP']-a['mAP'],NDS=c['NDS']-a['NDS'],
            tp_error_deltas={k:c[k]-a[k] for k in c if k not in ('mAP','NDS')},
            project_precision=p['aggregate']['C400']['primary']['precision']-p['aggregate'][name]['primary']['precision'],
            project_recall=p['aggregate']['C400']['primary']['recall']-p['aggregate'][name]['primary']['recall'])
    # Reference replacement is deliberately not automatic from two aggregate scores.
    better_b=changes['B']['mAP']>0 and changes['B']['NDS']>0
    better_a=changes['A']['mAP']>0 and changes['A']['NDS']>0
    overall='CandidateBenefitRequiresIndependentReplication' if better_a else 'MitigatedRelativeToB_AReferenceRetained' if better_b else 'NoConsistentBenefit_AReferenceRetained'
    acceptance=dict(stage='M8-J10',base_commit=cfg['base_commit'],engineering='Passed',training_steps=400,
        A_baseline_reused=True,B_forwards=0,B_updates=0,C_initialization='A_EmptyAdamW',
        same_J7_training_configuration=True,freeze_restore='Passed',precision='FP32',
        official_interface='PassedSameJ9Protocol',quality=overall,deltas=changes,
        final_candidate='C400',C200='RecordedOnly_NotSelected',reference='A',
        widespread_generalization='NotValidated',visual_driving_safety='NotValidated',
        next_training_authorized=False,budget_extended=False,main_merged=False,private_assets_uploaded=False,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    with Path('reports/m8j10_acceptance.json').open('x') as f:json.dump(acceptance,f,indent=2)
    lines=['# M8-J10：等400步多场景微调对照','',
        '本轮唯一新增训练C实际执行400次optimizer.step，从A和空AdamW开始；第400步检查点独立进程恢复通过。A复用J9，B只用历史缓存补官方评测，无B前向或参数更新。工程结果与质量结论分别记录。','',
        f'实际执行日期：{acceptance["completed_at_utc"][:10]} UTC；基础eb61246，训练执行提交b6627ba。最终汇总提交以Git记录为准。','',
        '## 固定训练与数据支持','',
        f'官方mini_train∩full train为6场景242帧，原始{data["totals"]["raw"]}条标注、任务过滤后8200条记录、558个独立保留instance。原八帧包含在内，其输入/监督逐数组完全一致；没有基于A/B表现删除困难样本，没有适配异常隔离。scene-0553/0796因不属于full train交集而排除。','',
        '六场景依次轮流，场景内独立固定种子洗牌、耗尽再洗牌。固定顺序、帧token和源时间戳见[manifest](m8j10_training_manifest.json)，真实曝光见[曝光统计](m8j10_exposure.json)。','',
        '|场景|池内帧数|训练GT记录|实际更新|曝光不同帧|','|---|---:|---:|---:|---:|']
    pre=read('reports/m8j10_preflight.json')
    for n,row in pre['scene_support'].items():
        x=e['per_scene'][n];lines.append(f'|{n}|{row["frames"]}|{row["GT_records"]}|{x["steps"]}|{x["unique_exposed_samples"]}|')
    lines+=['',f'400步覆盖全部{e["unique_exposed_samples"]}帧，曝光次数分布{e["exposure_histogram"]}。累计GT记录曝光C=13668、B=16750；步数相等但类别混合、目标密度和重复率不同，效果不能只归因于独立场景数量。连续帧记录不视为独立样本。','',
        'FP32/bs1/六摄512×1408/原十类，AdamW lr1e-5、betas(.9,.999)、wd0、clip35、无scheduler，与J7逐项相同。141个可训练Parameter，178个冻结Parameter和159个冻结buffer；backbone/neck eval，检测头train与原Dropout0.1。未修AMP、未换模型或解冻更多模块。','',
        '原checkpoint配置来自nuScenes full train，训练场景与其已知训练域重叠。mini_val与本次训练scene/sample均无交集，但早已用于开发比较，不是新的封存测试。原预训练逐样本账本/模型选择历史并未独立认证，沿用[J8来源限制](m8j8_pretraining_provenance.json)。','',
        '## A/B/C统一官方指标','',
        '本地nuScenes-devkit1.1.9、detection_cvpr_2019、相同81帧mini_val。Top300、score≥0.2、strict源XY范围、原类别半径、速度/属性规则不变；无NMS。官方GT3393，全部10类平均，无GT类别不删除。不是完整验证集、排行榜或论文可比成绩。','',
        '|对象|mAP|NDS|mATE m|mASE|mAOE rad|mAVE m/s|mAAE|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in scores.items():lines.append('|'+name+'|'+'|'.join(f'{row[k]:.6f}' for k in ('mAP','NDS','trans_err','scale_err','orient_err','vel_err','attr_err'))+'|')
    lines+=['','C200仅记录；C400是预声明最终对象，未用中间结果挑最佳checkpoint。A直接引用已验证J9指标。B/C均经同一个J9薄适配层和未修改的原官方导出器/评测器。完整分数与各类TP误差见[官方对照](m8j10_official_comparison.json)。','',
        '|类别|官方GT记录 / instance|A AP|B AP|C200 AP|C400 AP|','|---|---:|---:|---:|---:|---:|']
    for name in cfg['official_classes']:
        lines.append(f'|{name}|{o["GT_support"][name]} / {o["GT_unique_instances"][name]}|'+ '|'.join(f'{o["per_class"][m][name]:.6f}' for m in scores)+'|')
    lines+=['','无GT三类的AP=0与部分TP error=1是devkit汇总行为，不表示实测质量。cone/barrier不适用TP项在原始官方文件保持NaN，聚合副本明确为null；不当作0误差。bus只有一个独立实例等支持限制仍保留。','',
        '## 固定项目诊断','',
        'score≥0.2，同类一对一XY<2m，GT3559；官方3393分母不得混用。全部未匹配与重复预测保留。误差统计以已匹配目标为条件，同时报告召回与误检。','',
        '|对象|TP|FP|FN|重复|Precision|Recall|XY均值 m|XY P95 m|abs Δz均值 m|abs Δh均值 m|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in p['aggregate'].items():
        q=r['primary'];z=r['vertical']['all'];xy=q['matched_errors']['xy_m']
        lines.append(f'|{name}|{q["TP"]}|{q["FP"]}|{q["FN"]}|{q["duplicates"]}|{q["precision"]:.6f}|{q["recall"]:.6f}|{xy["mean"]:.6f}|{xy["p95"]:.6f}|{z["abs_dz_m"]["mean"]:.6f}|{z["abs_dh_m"]["mean"]:.6f}|')
    lines+=['','整体/逐类别/分场景/0.5m与1m/固定PR、竖直与尺寸尾部、z/h异常计数见[项目对照](m8j10_project_comparison.json)。A/B历史项目统计已核对精确一致，没有改旧工作点。','',
        '|场景|对象|TP|FP|FN|Precision|Recall|XY均值 m|','|---|---|---:|---:|---:|---:|---:|---:|']
    for scene,r in p['scenes'].items():
        for name,m in r['models'].items():
            q=m['primary'];lines.append(f'|{scene}|{name}|{q["TP"]}|{q["FP"]}|{q["FN"]}|{q["precision"]:.6f}|{q["recall"]:.6f}|{q["matched_errors"]["xy_m"]["mean"]:.6f}|')
    lines+=['','共同GT的配对比较避免把匹配集合变化误当定位改善。字段`*_A`表示相应参考A或B，`*_B`表示C400，仅为复用统计函数命名，不是再次训练B。','',
        '|比较|共同GT记录|参考独有|C独有|共同XY参考均值|共同XY C均值|平均差 C−参考|','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in p['comparisons'].items():
        q=r['aggregate'];er=q['paired_errors'];lines.append(f'|{name}|{q["common"]}|{q["A_only"]}|{q["B_only"]}|{er["xy_m_A"]["mean"]:.6f}|{er["xy_m_B"]["mean"]:.6f}|{er["xy_m_B_minus_A"]["mean"]:.6f}|')
    lines+=['','## 工程验证与权重','',
        f'真实更新恰好400次，逐步凭据完整；学习率唯一值{e["learning_rates"]}。冻结参数与buffer变化为0，141参数每步有有效梯度驱动更新，wd=0，不能以衰减充当学习。峰值每步分配{l["resources"]["peak_step_allocated_bytes"]/1024**3:.3f}GiB、reserved {l["resources"]["peak_step_reserved_bytes"]/1024**3:.3f}GiB；完整训练进程{l["resources"]["train_process_s"]:.1f}s，包括两次81帧评估/检查点保存。显存口径不含驱动及其他进程。','',
        '400步检查点在独立进程重建并校验模型、buffer、优化器、共享别名、RNG和固定eval输出；无额外更新。200/400都保留，不覆盖原始模型。','',
        '|对象|权重标识 / SHA256|','|---|---|',
        f'|A safe tensor|`{cfg["A"]["sha256"]}`|',f'|A原官方pth|`{cfg["A"]["original_pth_sha256"]}`|',f'|B400|`{cfg["B"]["sha256"]}`|']
    for step in (200,400):
        cp=read(base/'train'/f'checkpoint_{step:03}.json');lines.append(f'|C{step}|`{cp["sha256"]}`|')
    lines+=['','完整参数/恢复证据见[学习与冻结](m8j10_learning_and_freeze.json)，历史资产核验见[验收核验](m8j10_verification.json)。训练曲线是不同样本的逐步Loss，不据此宣称固定集合收敛；C200/C400固定开发集eval Loss单独记录。新增与复用runtime测试43项通过；主环境选帧/指标33项通过。初次环境变量缺失的skip保留记录，后续补齐环境检查全通过。','',
        '![聚合曲线](m8j10_figures/comparison.png)','',
        '## 结束决策','',f'工程：Passed。效果分类：`{overall}`。此自动分类只参考总体官方mAP/NDS，必须结合下面的实际收益和代价说明；不替代各类、场景和定位维度。','',
        '本轮到400步结束，不追加训练、不调阈值、不改模型、不启动CARLA。单种子、两个高度相关开发场景不支持统计性或安全结论。即使出现退化，也不能证明更大数据或PETR架构无效。','',
        '下一步只建议一个限定实验：在保持C样本池、400步预算与配置不变的前提下，另行批准第二个固定种子重复，以检验本轮方向是否稳定；本轮不执行。A继续保留为参考，C400作为独立候选研究资产。','',
        '复现命令、固定协议和隐私边界见[实现说明](../docs/M8J10_IMPLEMENTATION.md)。未上传原图、逐对象输出、标签或权重，未合并main。']
    with Path('reports/m8j10_summary.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
