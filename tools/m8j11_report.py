"""Aggregate-only final report. Never publishes object records, images or weights."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(Path(p).read_text())
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def write(p,x):
    with Path(p).open('x') as f:json.dump(clean(x),f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    c=read('configs/m8j11_protocol.json');A=read('reports/m8j9_sources/resume/eval2/metrics_summary.json');S=read('reports/m8j11_sources/metrics_summary.json');q=read('reports/m8j11_project_comparison.json');support=read('reports/m8j11_sources/official_filters.json')
    ar=read('reports/m8j11_sources/A_raw_timing.json');sr=read('reports/m8j11_sources/StreamPETR_raw_timing.json');full=read('reports/m8j11_sources/full/result.json');probe=read('reports/m8j11_sources/probe/result.json');verify=read('reports/m8j11_verification.json')
    assert full['status']==probe['status']=='ReadOnlySequencePassed' and verify['status']=='CacheSourceAndOfficialReproducibilityPassed'
    official=dict(name='冻结A与冻结StreamPETR在mini_val上的本地官方devkit评测，使用指定缓存与输出策略',
        A=A,StreamPETR=S,delta_Stream_minus_A=dict(mAP=S['mean_ap']-A['mean_ap'],NDS=S['nd_score']-A['nd_score'],TP_errors={k:S['tp_errors'][k]-A['tp_errors'][k] for k in A['tp_errors']}),
        GT=support['official_GT_filtered'],unique_instances_by_class=support['GT_instances_by_class'],official_zero_support_classes_retained=True,
        devkit='1.1.9',config='detection_cvpr_2019',score_threshold=.2,TopK=300,NMS=False,scenes=2,frames=81,official_GT=3393,project_GT=3559,
        warning='Development comparison; not leaderboard/full validation/temporal causal ablation/safety result')
    write('reports/m8j11_official_comparison.json',official)
    initial=Path('reports/m8j11_resource_comparison.json');initial.rename('reports/m8j11_sources/initial_instrumented_resource_summary.json')
    resource=dict(A=ar,StreamPETR=sr,precision=dict(A='FP32',StreamPETR='FP32 image/model compute; official FP64 timestamps'),
        note='Primary same-machine sequential raw RGB boundary, no concurrent compilation/training. Includes first process/scene frames. Cached-input and instrumented diagnostic timings separately retained, not mixed into primary table.',
        excluded='One-time model/metadata loading, GT scoring, file serialization, state audits; not sensor capture/planning/control.',
        cached_input_A=read('reports/m8j11_sources/A_timing_uninstrumented.json')['summary'],
        candidate_timing_repeat='reports/m8j11_sources/timing/result.json',sample_limit='81 correlated frames, two scenes; no worst-case or strict realtime claim')
    write(initial,resource)
    evidence=dict(inspection='Assistant visual inspection, not human reviewer signoff',viewed=['00_six.jpg','40_six.jpg'],
        generated_indices=[0,20,39,40,60,80],directory=c['private_root']+'/visuals',
        observations=['Vehicle/person projections visibly align in multiple views; substantial duplicate/spurious predictions remain for both models.','Only two mosaics viewed; other generated frames are not marked reviewed.'],human_signoff='NotRequested_NotClaimed')
    write('reports/m8j11_private_visual_review.json',evidence)
    decision=dict(engineering='FrozenCandidateLocalComparisonPassed',quality='CandidateResearchGo_AReferenceRetained',
        pure_visual_driving_safety='NotValidated',default_model_replaced=False,temporal_causal_gain='NotEstablished',broad_generalization='NotEstablished',
        runtime='MeasuredOfflineOnly_NotRealtimePassed',optimizer_steps=0,backward_calls=0,CARLA=False,training=False,
        source_commit=c['source_commit'],checkpoint_sha256=c['initial_asset_sha256'],execution_commit=full['code_commit'],base_commit=c['base_commit'],
        actual_date_UTC=full['started_utc'],evidence_classes=dict(quality='ActualRTX4090Forward_OfflineOfficialAndProjectEvaluation',tests='SyntheticCPUAndCUDA',temporal='ActualShortSequenceAndFullRun',safety='NotRun'),
        tests=dict(isolated=11,passed=11,failed_final=0,initial_failures_preserved=True),
        state_integrity=True,scene_resets=True,independent_repeat_equal=True,GT_and_pointcloud_input=False,
        next_step='One separately authorized frozen-weight temporal robustness pilot on already-local sequences: dropped/delayed/reordered timestamp and reset handling, with A as reference. No training or default replacement now.',
        residual_risks=['424 project FN,3277 FP,793 duplicate predictions remain','Three official classes have no support; bus only one instance','Public config supports full-train origin but checkpoint has no per-sample ledger','Different resolution/backbone/training schedule/backend/ego-motion inputs confound temporal attribution','Raw RGB wallclock P95 exceeds200ms; no runtime gate closure'])
    write('reports/m8j11_acceptance.json',decision)
    rows=[]
    for n,m in [('A',A),('StreamPETR',S)]:rows.append('|'+n+'|'+ '|'.join(f'{x:.6f}' for x in [m['mean_ap'],m['nd_score']]+[m['tp_errors'][k] for k in ('trans_err','scale_err','orient_err','vel_err','attr_err')])+'|')
    clsrows=[]
    for name in c['classes']:
        clsrows.append(f"|{name}|{support['official_GT_filtered']['classes'][name]} / {support['GT_instances_by_class'][name]}|{A['mean_dist_aps'][name]:.6f}|{S['mean_dist_aps'][name]:.6f}|")
    projrows=[]
    for n in ('A','StreamPETR'):
        x=q[n]['primary'];v=q[n]['vertical']['all']
        projrows.append(f"|{n}|{x['TP']}|{x['FP']}|{x['FN']}|{x['duplicates']}|{x['precision']:.4%}|{x['recall']:.4%}|{x['matched_errors']['xy_m']['mean']:.4f}|{x['matched_errors']['xy_m']['p95']:.4f}|{v['abs_dz_m']['mean']:.4f}|{v['abs_dh_m']['mean']:.4f}|")
    resrows=[]
    for n,r in [('A',ar),('StreamPETR',sr)]:
        x=r['summary']['all'];g=x['gpu_ms'];w=x['raw_to_cpu_prediction_ms']
        resrows.append(f"|{n}|{g['median']:.2f}/{g['p95']:.2f}/{g['max']:.2f}|{w['median']:.2f}/{w['p95']:.2f}/{w['max']:.2f}|{x['peak_allocated_bytes']['max']/2**30:.3f}|{x['peak_reserved_bytes']['max']/2**30:.3f}|")
    text='''# M8-J11：冻结权重的首个感知候选对照

已完成官方单一 StreamPETR-R50 候选的隔离环境、短连续序列检查及全部81帧只读对照。**候选值得进入下一轮研究，PETR-A继续作为参考，默认模型不替换。** 实际执行日期2026-10-29 UTC；基础68ee3e8，独立分支feat/m8j11-streampetr-r50-frozen-candidate。未训练、未启动CARLA，未修改历史资产或合并main。

## 固定对象与工程差异

|项目|PETR-A|StreamPETR候选|
|---|---|---|
|主干|ResNet50 Caffe风格，stage3/4 DCNv2，C5|ResNet50 PyTorch风格，无DCN，CPFPN/stride16|
|六摄输入|512×1408，BGR均值、std1|256×704，RGB ImageNet均值/std|
|Query / 参数量|900 / 35,805,405|644+256传播 / 37,259,345|
|训练来源|公开nuScenes train配置24epoch、MSRA R50|官方ImageNet R50、nuScenes train配置90epoch|
|时间输入|单帧|历史预测记忆1024、每帧写入256，真实时间戳与标定位姿|
|推理|PyTorch2.0/cu118、MMCV1.7.2、MMDet3D0.17.1|隔离PyTorch2.0/cu118、MMCV1.7.0、MMDet3D1.0.0rc6|
|精度|FP32，TF32关闭|实际卷积/Linear/Attention和输出FP32，官方时间戳FP64，TF32关闭|

唯一候选配置 `stream_petr_r50_flash_704_bs2_seq_90e.py`；官方源码 `95f64702306ccdb7a78889578b2a55b5deb35b2a`。权重154852451bytes，SHA256 `e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451`。A原pth SHA `16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad`。本地哈希用于固定文件，官方没有给出独立预期digest。

官方说明Flash可选，实际使用该仓库的非Flash FP32 Attention；591个权重张量严格加载，无键缺失/多余。仅私有源副本跳过可选Flash/EVA导入。没有更换权重或依据mini_val选配置。官方来源：[固定仓库](https://github.com/exiawsh/StreamPETR/tree/95f64702306ccdb7a78889578b2a55b5deb35b2a)、[选定权重](https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_90e.pth)。

预训练来源由官方配置/模型表支持，候选文件仅state_dict，没有可独立认证的逐样本训练账本。mini_val属于官方full-val，但已多次用于本项目开发；不能称为未使用过的盲测。输入分辨率、主干算子、训练时长、时序信息和依赖均不同，这是现成配置工程对照，不是时序单变量消融。

## 输入、时序、导出验证

沿用81帧完整mini_val清单，40+41帧，按scene和真实时间戳顺序执行，每个场景从第一帧开始。图像文件哈希与历史清单全部相同。各相机使用自身标定与采集位姿；官方命名ego_pose实际为LIDAR_TOP坐标到global的变换，只读元数据，不读取点云/Radar/GT特征。

官方resize0.44/crop(0,140,704,396)和归一化原样复用。新rc6的bottomXYZ/LWH/CCW yaw转换为J9的bottomXYZ/WLH/SECOND yaw，仅用于预测导出；八角点及独立全局变换检查通过。Top300原生截断后应用固定score≥0.2、strictXY51.2和原类别半径，无NMS、无GT修正。内部记忆写入仍发生在公共过滤前。J9的use_external=False输出metadata沿用，但ImageNet预训练事实在此披露，该标志不参与指标计算。

短序列12次前向包含重复、跨场景重置和第二场景独立冷启动，最大重复差异0。正式81帧使用新进程、新记忆；两场景首帧均计入。5个重置探针与正式缓存逐数组完全一致。参数/注册buffer全程哈希不变，只有合法时序记忆更新（写入后1280，下一帧先截取1024）。无optimizer/backward。11项独立测试通过，官方同文件重复评测summary/details除eval_time完全一致。

初次输入检查错误地要求官方float32矩阵满足1e-8，已保留失败。已定位最大外参数值差0.0001081、pose逆积差2.384e-7，来自官方FP32与千米级global坐标消减；使用由FP32 epsilon和矩阵规模导出的舍入界核对。修复仅检查器，旧/新648个输入数组完全相同，没有改标定或模型。

## 官方本地mini_val结果

名称：**冻结A与冻结StreamPETR在mini_val上的本地官方devkit评测，使用指定缓存与输出策略**。nuScenes-devkit1.1.9 / detection_cvpr_2019，官方GT3393。不是排行榜、完整验证集或论文可比分数。

|模型|mAP|NDS|mATE m|mASE|mAOE rad|mAVE m/s|mAAE|
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

|类别|官方GT记录 / instance|A AP|候选AP|
|---|---:|---:|---:|
'''+ '\n'.join(clsrows)+'''

无支持的construction_vehicle/trailer/barrier保留官方平均行为，不能解释为实测质量；官方不适用误差NaN保留在原文件，聚合JSON用null。bus仅1个独立instance；连续帧GT记录不是独立目标。全部TP误差及每类细项见[官方对照](m8j11_official_comparison.json)。候选所有有支持类别AP提高，但并非每个维度都改善：bicycle mASE约+0.0148、mAOE约+0.1377rad，bus mAAE约+0.0244。

## 项目固定工作点

项目GT3559，不能和官方3393混用；同类一对一XY<2m，阈值0.2。以下误差仅针对匹配目标，同时保留漏检/误检。

|模型|TP|FP|FN|重复|Precision|Recall|XY均值m|XY P95m|absΔz均值m|absΔh均值m|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(projrows)+'''

两场景均改善：scene-0103的TP1353→1407、FP3476→1969、Recall83.47%→86.80%；scene-0916的TP1594→1728、FP2222→1308、Recall82.25%→89.16%。共同匹配2738条，A-only209、候选-only397。共同目标XY平均0.8281→0.5613m；1947条改善、791条退化，不能称为逐目标一致改善。

匹配记录absΔz>1m从1→0，absΔh>1m均0；这是有限样本描述性异常统计，不是安全阈值通过。完整0.5m/1m/2m、固定PR、分类别/分场景和z/h尾部见[项目对照](m8j11_project_comparison.json)。仍有424条FN、3277条FP、793个重复，可靠障碍感知和驾驶安全门禁不关闭。

## 同机资源与时效

主要资源表使用顺序、非并发的81帧原始RGB路径：本地JPEG读取/解码、官方resize/normalize、标定变换、H2D、原模型及原生解码、D2H。元数据和模型一次性加载、GT评价、检查和落盘在计时外。GPU列是CUDA Event围住模型/原生解码的时间区间，含该区间调度空隙，不冒称纯kernel忙碌时间。所有计时结果与固定质量缓存相符，没有用计时结果重新选预测。

|模型|GPU P50/P95/Max ms|完整本地处理P50/P95/Max ms|峰值allocated GiB|峰值reserved GiB|
|---|---:|---:|---:|---:|
'''+ '\n'.join(resrows)+'''

候选进程首帧GPU499.18ms/总761.90ms；第二场景冷记忆首帧158.90/403.42ms。A相应为879.93/1293.95和521.00/919.65ms；A无时序记忆，“第二场景冷”仅为对齐分组。两个冷帧不足以估计冷启动分布。allocated/reserved为PyTorch进程统计，不含驱动/桌面/其他进程，也不是训练显存。

候选GPU P95约160ms，但含图像预处理的P95约419ms，超过200ms。81帧不能证明最坏时延，未验证共享GPU、在线摄像头或控制链，**Runtime保持未通过**。初次含诊断同步/同时CPU编译的测量保留，不用于主要资源表。未启用AMP/TF32/TensorRT/Flash加速。

## 交付和决策

- 4090可运行：本轮冻结候选FP32工程接口通过，非训练验证。
- 质量与资源：总体及两场景有收益，同时保留类别误差退化、漏检/重复和时效不足。
- 候选资格：值得保留为下一轮研究候选；A继续参考，默认模型不替换。
- 因果限制：未证明独立时序收益、广泛泛化或驾驶安全。官方预训练/模型选择历史并非完整可审计。

下一步只建议一个另行授权的**冻结候选时序鲁棒性小实验**：在已有本地片段上测试帧缺失/延迟、时间戳乱序拒绝和显式重置对检测的影响，A作为参考；不训练、不扩数据、不接控制。本轮到此停止。

私有对照：`/root/gpufree-data/hs-drive-private/m8j11-streampetr/visuals/index.html`，固定两场景首/中/末帧。助手实际查看00/40六摄拼图，仅作投影诊断，不伪称人工签核或全量审查。原图、逐对象GT/预测、权重均不上传。

[验收](m8j11_acceptance.json) · [复现核验](m8j11_verification.json) · [资源明细](m8j11_resource_comparison.json) · [实现及命令](../docs/M8J11_IMPLEMENTATION.md)。各执行日志/退出码/源码哈希位于m8j11_sources。初始网络失败、误用全局Git镜像的隔离记录、版本冲突、构建超时、检查器失败全部保留；最终未以删除失败获得通过。
'''
    Path('reports/m8j11_summary.md').write_text(text)
    fig,axes=plt.subplots(1,3,figsize=(14,4));names=['A','StreamPETR'];xx=np.arange(2)
    axes[0].bar(xx-.17,[A['mean_ap'],S['mean_ap']],.34,label='mAP');axes[0].bar(xx+.17,[A['nd_score'],S['nd_score']],.34,label='NDS');axes[0].set_xticks(xx,names);axes[0].legend();axes[0].set_title('Local official mini_val, fixed output')
    axes[1].bar(xx-.17,[q[n]['primary']['precision'] for n in names],.34,label='Precision');axes[1].bar(xx+.17,[q[n]['primary']['recall'] for n in names],.34,label='Recall');axes[1].set_xticks(xx,names);axes[1].legend();axes[1].set_title('Project 2m / score0.2')
    axes[2].bar(xx,[ar['summary']['all']['raw_to_cpu_prediction_ms']['p95'],sr['summary']['all']['raw_to_cpu_prediction_ms']['p95']]);axes[2].set_xticks(xx,names);axes[2].set_ylabel('ms');axes[2].axhline(200,color='red',linestyle='--');axes[2].set_title('Raw RGB processing P95')
    fig.tight_layout();d=Path('reports/m8j11_figures');d.mkdir(exist_ok=False);fig.savefig(d/'aggregate_comparison.png',dpi=160);plt.close(fig)
    print('Final aggregate reports written')
if __name__=='__main__':main()
