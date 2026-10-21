# M5-L 实施与复现

基线`0175f21`，独立分支`feat/m5l-sparse-geometry-feasibility`。所有新资产前缀m5l，旧模型/标签/报告只读。工具拒绝覆盖，复现请使用独立工作副本和新输出路径，不能删除旧失败记录。

## 环境与范围

RTX4090 24GB，PyTorch2.8/CUDA12.8，Python3.10。GPU使用`.venv-gpu/bin/python`；CARLA SDK使用`.venv/bin/python`。在线Shadow主进程显式加载SDK路径，子进程不导入CARLA。只读离线研究和静态Shadow，不执行车辆控制。

## 现有312帧诊断与有限训练

```bash
.venv-gpu/bin/python -m tools.m5l_prepare
.venv/bin/python -m tools.m5l_sampling_audit
.venv-gpu/bin/python -m tools.m5l_p4_probe
.venv-gpu/bin/python -m tools.m5l_vehicle_train
.venv-gpu/bin/python -m tools.m5l_vehicle_eval --split val
.venv-gpu/bin/python -m tools.m5l_object_eval --split val
.venv-gpu/bin/python -m tools.m5l_adaptive --split val
.venv-gpu/bin/python -m tools.m5l_development
```

实际Vehicle-only训练输出`runs/m5l_residual_v2`。首次脚本在写样本顺序时中止，未执行优化，原目录与日志保留。B/C同初始化、同容量、同顺序，各600步两种子。原网络冻结，只优化M5-K零初始化残差；M5-L新协议限定车辆，不改变旧门禁。其余前景不进入优化，近场Actor质量仍单独评估。

Loss：分近/远vehicle/background平衡CE +0.25候选存在的vehicle同层1m质量NLL +0.5M4近场Replay CE +0.1新近场与Replay KL。原M4 Replay只用train优化；val只读。Replay无Mobile缓存，q=0；真实Mobile近场保护依赖新Val单列检查，不能只用零q Replay宣称完整路径保护。Val在0/200/400/600选模型，不按Test增训。

P4采样支持不达标，未训练P4Depth。实际P4特征探针使用Mobile浅层S4 layer3的24通道与训练过P8语义64通道局部融合；卷积中心与历史Depth名义中心分别定义。测量2880个合法固定局部点，非全六摄P4 Dense Splat。C路径未训练，质量记NotEvaluated。

2.5D由RGB二值前景连通域产生候选，固定平面z=-0.25m、K/T反投影，边界/地平线/敏感性拒绝。无GT推理输入。其余输出仍是contact-unverified条件假设，无法自动认定底部接地或没有遮挡。评测匹配只使用GT可见关联mask，报告候选合并/分裂、未匹配和abstain；XY中心为AABB代理，可见底部端点另列，不称完整3D框。

## 新封存测试与在线Shadow

先完成现有312帧全部分析，再运行：

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5l_capture_protocol
.venv-gpu/bin/python -m tools.m5l_online_shadow
.venv-gpu/bin/python -m tools.m5l_freeze
.venv-gpu/bin/python -m tools.m5l_test_pipeline
.venv-gpu/bin/python -m tools.m5l_finalize
```

`m5l_postdev`可顺序运行Shadow→freeze→test→finalize，避免重复运行同输出。新测试4地图位置、19配置/位置；旧M5-K Test标regression。Town02隔离候选不足，失败记录保留，最终分配Town01/03/04/05，45m以上隔离；坡道名义pitch可能超过360度，实际角度由三角函数/pose矩阵解算，不按原始大角度宣称坡度。

Shadow唯一TickOwner在主进程；worker只能接收六摄完整buffer，不取得world、不调用tick/control。2Hz按墙钟调度，队列容量1、只保留最新，原sim时间戳不改。每包记录首次callback墙钟、queue年龄、转换H2D、推理、输出及完整墙钟；不是从渲染开始计时。基线/并发/恢复各20warmup+240ticks，超出预声明长尾限即停。静态场景不验证驾驶负载下安全或实时性。

## 测试、审计和原始资产

```bash
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_imaging_rig.py tests/test_snapshot.py tests/test_session.py tests/test_geometry_extended.py
```

`runs/m5l_logs/`保存全部原始日志，`runs/m5l_*`保存逐Actor/逐帧数据与Checkpoint。新原始CARLA数据在`/root/gpufree-data/hs-drive-data/raw/m5l_test`，监督分别在processed sidecar与pixel路径。GT不进入输入合同。精简报告引用完整SHA索引，大文件不提交Git。
