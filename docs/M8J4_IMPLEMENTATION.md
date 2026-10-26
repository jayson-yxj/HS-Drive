# M8-J4 实现、合同与复现

基线 86984a6843a0ebbc49f1e90a97afac6c509515fd；执行于 2026-10-26。只使用已解压 M8-J3 单片段与已有 PETR 隔离环境。配置在 configs/m8j4_protocol.json；所有结果版本均保留，不改旧默认模块。

## 模块与监督合同

| 模块 | 责任 |
|---|---|
| datasets/m8j4_gt.py | Box 转换、近面裁剪、图像投影、Instance 区域证据及一对一关联限制 |
| tools/m8j4_sidecar.py | 214 组原始独立 Sidecar 和私有图像；不运行模型 |
| tools/m8j4_resolve_review.py | 将跨帧/跨相机身份歧义降级为 Unknown，追加 reviewed_sidecar |
| tools/m8j4_render_review.py | 重现隔离歧义后的私有 RGB/Instance 拼图；本轮图片最初以等价的本地 CPU 调用生成 |
| tools/m8j4_final_audit.py | 23 条异常的逐视图像素审核、范围/裁剪支持统计 |
| datasets/m8j4_petr_input.py | 只接收 RGB/K/T；保留官方预处理，显式传播 K 和裁剪 |
| tools/m8j4_prepare_forward.py | 复用 J3 RGB-only 加载器，生成只有三种数组的数值 NPZ |
| tools/m8j4_run_forward.py / m8j4_forward.py | 隔离环境、无网络、冻结权重三次前向及实际 GPU 记录 |
| tools/m8j4_compare.py | 推理完成后读取监督，生成私有比较图及汇总 |
| tools/m8j4_package_evidence.py | 汇总证据、私有 SHA 清单及验收；不修改实际运行记录 |

以 S=diag(1,−1,1,1) 将 CARLA Ego（x 前、y 右、z 上）转换为 HS Ego（x 前、y 左、z 上）：p_HS=S·world2ego·p_world。Box 参考始终为 Actor Box Center，不是轮胎接地点或精确 Mesh。尺寸是完整 length/width/height，不是 extent；原八角点完整保留。由固定角点边恢复轴方向，并检查中心、尺寸及正交残差，Yaw=atan2(前向轴 y, 前向轴 x)，以弧度逆时针计。倾斜大于 2° 的记录保留完整几何，但不进入 yaw-only 正样本索引。

相机投影采用 optical_from_CARLA_cam·world2cam。跨近面的 Box 按 12 条边与 z=0.1m 截交后取投影凸包，再裁剪到图像范围；不把负深度角点直接投影。近面相交、遮挡比例不明、无渲染支持等保持 Unknown。Behind/Outside 的 view Invalid 只限定该视图，不自动否认其他视角下的目标。

源 Instance PNG 为 BGRA。语义 R 与 Semantic 图交叉核对，G+256B 只是截断的 UE render token。关联使用投影凸包覆盖、区域 IoU、语义一致性、候选间差值及唯一性；阈值先写入 protocol。对同视角多 Actor 共享区域、跨帧多 Actor 竞争相同 token 等保守拒绝。轮廓面积/投影凸包面积只是遮挡提示，不能充当真实可见比例。

Valid 表示自动证据支持；Invalid 表示明确不适合作该视图或目标监督；Unknown 表示证据不足。三者分开保存。所有记录 training_negative=false，所有帧 background_negative_valid=false、occupancy_free_valid=false。Sidecar 不是可直接供原 PETR 完整分类损失使用的穷尽标注。

RGB-only 数值输入只允许 images、intrinsics、cam_to_ego。来源是未经改动的 J3 allowlist 加载器，它已经检查图像 SHA、相机顺序、K/T 及未知时间语义。GT 不进入 NPZ、模型元数据或特征。框架字段 lidar2img 只是既有投影矩阵名称，本次没有 LiDAR 输入。原始 source_frame/timestamp_s 继续为 null，temporal_valid=false；不得把 ordinal×固定步长当作真实时间。

## 前向预处理与执行

Bench2Drive 的六摄 rig 及 FOV 与 HS 原 rig 不同。实际图像为 900×1600，保留实际 K/T；官方 PETR 测试 resize=0.88，得到 792×1408，裁剪 (0,280,1408,792)。图像矩阵 A=[[0.88,0,0],[0,0.88,−280],[0,0,1]]，K'=AK。官方 BGR 归一化、32 倍数 padding、测试无 flip 保持不变。

数值入口 [1,6,3,512,1408]；现有 PETR 原地 squeeze batch=1 张量，所以旧执行 JSON 的 shapes.input 是调用后的 [6,3,512,1408]。m8j4_model_forward.json 追加解释并保存原模型源码 SHA，未回写实际输出。框架底中心/w-l-h/yaw 与 HS 中心/l-w-h/CCW Yaw 的转换复用已有 m8b_petr_adapter，实际解码角点交叉检查通过。

所有 CUDA 时间先后同步；第一次前向包括冷启动，未另加预热。峰值为 PyTorch allocator allocated/reserved，不等于整个显卡用量。三帧结果只证明接口可运行，不能形成可靠性能分位数或检测质量结论。

## 已执行命令

以下在仓库根目录执行。先完成几何和关联门禁，再准备模型输入。初次输入门禁因身份冲突失败，见 reports/m8j4_sources/input_gate_initial_failed.log；隔离歧义后重新执行 CPU 门禁，只有一次 GPU 三帧作业。

```bash
.venv/bin/python -m tools.m8j4_sidecar
.venv/bin/python -m tools.m8j4_resolve_review
.venv/bin/python -m tools.m8j4_render_review
.venv/bin/python -m tools.m8j4_prepare_forward
.venv/bin/python -m tools.m8j4_run_forward
.venv/bin/python -m tools.m8j4_compare
.venv/bin/python -m tools.m8j4_final_audit
.venv/bin/python -m pytest tests/test_m8j4_gt.py tests/test_m8j4_contract.py tests/test_m8j3_rgb_clip.py -q
.venv/bin/python -m tools.m8j4_package_evidence
```

这些生成器采用 exclusive-create，不覆盖旧结果。**不要在当前有产物的目录直接重复运行。** 重现时使用本分支的独立 worktree、同一已授权本地载荷/现有运行环境，将 protocol.private_root 指向新的私有空目录。m8j4_forward 通过环境参数读取新目录；其余模块读取 protocol。无需重新下载数据或模型。

以下是可执行的重现目录准备方法：只移动新 worktree 中的 M8-J4 报告副本到保留目录，不操作本次原始报告。为避免误运行，固定目录已存在时命令拒绝继续；独立重现会再次执行三帧前向，应由下一次需要复现时显式执行。

```bash
git worktree add --detach /root/gpufree-data/hs-drive-experiments/m8j4-reproduce feat/m8j4-single-frame-gt-forward
cd /root/gpufree-data/hs-drive-experiments/m8j4-reproduce
/root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/.venv/bin/python - <<'PY'
import json, shutil
from pathlib import Path
archive = Path('reports/m8j4_original_evidence_copy')
archive.mkdir(exist_ok=False)
for p in list(Path('reports').glob('m8j4_*')):
    if p != archive:
        shutil.move(str(p), str(archive / p.name))
src = Path('reports/m8j4_sources')
src.mkdir()
for p in (archive / 'm8j4_sources').glob('*.cpp*'):
    shutil.copy2(p, src / p.name)
cfgpath = Path('configs/m8j4_protocol.json')
cfg = json.loads(cfgpath.read_text())
cfg['private_root'] = '/root/gpufree-data/hs-drive-private/m8j4-reproduce'
assert not Path(cfg['private_root']).exists()
cfgpath.write_text(json.dumps(cfg, indent=2))
PY
```

此后按上面模块顺序执行，将 .venv/bin/python 替换为 /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/.venv/bin/python。测试输出保存为 reports/m8j4_sources/tests_final.log，才能运行 package_evidence。package_evidence 中图像检查记录是本次人工输入的助手观察：重现时必须重新检查图片并填写新的观察，不能把本次观察转成新的签核。

已有 private_sidecar_invariants 测试明确对本次固定私有路径检查；没有私有数据的环境会 Skip，该情况不能声称复现了真实载荷审核。本次运行没有 Skip。源码中的阈值/统计依赖本次单片段，不是通用数据集认证工具。

仅需要重查合同时可安全运行上面的 pytest 命令。实际测试覆盖中心/镜像/Yaw/角点、畸形 Box、近面/背面/图像外、render token 编码和语义错误、共享区域歧义、输入 K/T/裁剪、监督隔离、214 帧真实状态不变量及正样本范围限制。18 项全部通过。

## 私有查看与发布边界

私有根目录为 /root/gpufree-data/hs-drive-private/m8j4-single-frame。

- sidecar/：首次自动结果，保留原冲突。
- reviewed_sidecar/：撤回歧义后的版本；文件 ordinal 不等于原始 Frame。
- reviewed_visuals/：53 帧六摄 RGB 和 Instance 拼图；uniform_00..05 覆盖 24 个均匀帧，anomalies_00..05 覆盖全部 23 个异常帧。
- anomaly_detailed_audit.json：23 条记录的原坐标、六摄像素数及排除原因。
- inference_input/：三帧 RGB/K/T 数值输入，无 GT。
- forward/：完整 900 条预测、原始运行统计及未改写的形状日志。
- comparison/：三帧 BEV 对照和描述性最近邻诊断；不是一对一匹配或 AP。
- artifact_inventory.json：私有全量文件 SHA 清单；其 SHA 在公开汇总中。

本轮查看 44 个不同帧的六摄 RGB 概览和 3 个 Instance 拼图，另查看 00107 的 BEV 对照。未声称检查每一个全分辨率像素；自动证据支持的对象仍需人工审核，human signoff 保持 Pending。实际检查记录见 m8j4_visual_review.json。

Git 只包含代码、配置、聚合统计、必要官方源码引用及执行/测试日志。不得加入原始/提取图像、逐对象 Sidecar、预测数组或 Checkpoint。历史报告、默认驾驶模块和 main 均不变。

## 最小训练实验尚需完成

本轮不实现训练损失、不更新权重。后续首先选择人工确认的 1–4 帧；明确可监督正目标及未知区域，不允许标准匹配器将未知 Actor/Query 自动标为背景。速度、uint8 Depth、expert_assessment 与未证实的 Occupancy Free 均不得参与损失。检查 GT 不进入推理，保留原始 K/T 与图像变换，冻结 Backbone 并预声明有限预算、梯度稳定性与类别坍缩验收。此项仅允许检查学习管线，不能宣布独立场景泛化或视觉控制通过。
