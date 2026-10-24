# M8-B Frozen PETR CARLA Adapter & Zero-Shot Quality Pilot

本轮基于 `a3a892219e29c3c16d969e8b0d934173a6f8dbb0`，独立分支 `feat/m8b-petr-carla-zero-shot`。实际服务器 RTX4090 执行192帧冻结FP32推理；没有训练、采集新CARLA帧、修改默认模型或调用车辆控制。质量证据来自历史记录回放，不是车辆闭环。

## 输入、坐标与隔离

`datasets/m8b_petr_adapter.py:PETRCarlaOfflineAdapter` 接收严格字段白名单 packet。准备工具读取原始档案、审计、复制六张RGB至独立目录，导出K、Camera-to-Ego、frame与timestamp。`evaluation_index.json` 单独保存GT索引，推理manifest中没有sample/split/Actor/GT路径。推理器只读这个清洁目录和M8-A数值权重。

相机顺序为front/front_left/front_right/rear_left/rear_right/rear；分别映射CAM_FRONT/FRONT_LEFT/FRONT_RIGHT/BACK_LEFT/BACK_RIGHT/BACK。检查六摄同帧同时间、SHA、图片尺寸、唯一相机、右手刚体外参、光轴与相机名称匹配。标定跟随图像，不使用文件排序。名义光轴检查不能单凭像素证明物理传感器身份；原始导出SHA提供来源链。

使用官方 `ResizeCropFlipImage(training=False)`，实际源尺寸384×640，缩放2.2，整数目标844×1408，裁剪(0,332,1408,844)。RGB→BGR，mean=[103.530,116.280,123.675]，std=1，to_rgb=False，pad divisor32。保存A、K_aug、lidar2img及SHA。`K_aug=A K`；官方名义缩放与整数resize高度存在0.8px舍入，不私自改变实现。按本次样本逐相机验证正负光学深度和逆投影。

HS参考x前/y左/z上；optical x右/y下/z前。`lidar2img=K_aug4 inverse(T_cam_to_ego)`，不消费点云或世界GT位姿。官方旧mmdet3d 0.17 box底面中心转几何中心z+=h/2，w/l/h→l/w/h，yaw_HS=wrap(-yaw_old-π/2)，vx/vy是参考坐标分量无需随boxyaw旋转。非方形box与官方corners比较，最大约微米量级；坡道参考OBB无法完全用upright yaw表达，单独量化角点误差，不能称精确mesh。

`unshare --net` 禁用推理网络；清理环境变量、GT路径Python open审计、两个拒绝canary，使用已解包且SHA固定的numeric NPZ，allow_pickle=False。**这是可信冻结代码下的进程隔离/路径审计，并非防恶意native代码的完整文件系统沙箱。** GT进程与推理进程分离；未导入Evaluator。权重严格558张量加载、前后逐位对比，requires_grad=False/no_grad，0训练步。最后增加的非有限timestamp拒绝检查仅验证合同，不改变此次已执行有限输入的张量；执行时源码SHA保存在GPU日志JSON。

## 固定数据与评价

`configs/m8b/protocol.json` 在推理前固定规则。完整40份M7-A输入中38份没有已记录可见目标，因此按38个M5-O位置组中eligible帧configuration顺序的0.2/0.4/0.6/0.8分位补充，去重后192帧。Train102/20组，Val30/6组，Calibration30/6组，旧Sealed30/6组。所有样本都保留；没有按分数筛样本。core40先完成并落盘，再处理补充样本。按冻结规则标识的smoke覆盖见dataset_support，含Town01/Town03、各相机、Vehicle/Person/Cone、近中远距离。

Evaluator按历史M5-O seed101的可靠Actor-view记录去重成帧/Actor分母，共155目标、265视角、119组内Actor身份。实例ID几何关联+有限Depth[1.5,80)且>=3像素，与旧缓存像素数逐项一致；这是渲染器关联/OBB参考，非独立mesh真值。标注不覆盖所有真实背景。38空场景仍有道路设施；“empty”不是物理无物体。

1. 自然检测：保留官方全部300 topK。score>=0.1，粗类一致，XY<=2m且|Z|<=2m，分数降序一对一；4m敏感性另报。匹配的是中心门限，**不是IoU检测AP**。所有候选计入负担，未匹配FP仅表示相对记录标注未匹配。
2. 条件定位：在有可靠且裁后仍可见像素的视角，将GT与预测OBB投影，近面裁边后算图像矩形IoU；同类、IoU>=0.1，最大匹配数后最大总IoU一对一，**不按中心误差过滤**。它是关联代理，重叠或相似背景可能错误配对，不能声称每个关联都确认物理身份。自然2m匹配子集也另报，但其尾部已被截断。
3. 每个阈值重新做一对一关联，所以条件曲线可能非单调；它不是固定对象集合的严格校准函数。置信分数不是安全概率。10个阈值预先声明，只以独立Calibration资格门槛选择；本轮无合格阈值。
4. 裁剪统计使用native像素中心v*2.2>=332；插值边界约1px不确定。Ground/Unknown不进入物体类分母，无输出一律不是Free。
5. 基线用历史seed101/102、100%数据模型的冻结预测。自然2D连通域检测与PETR3D检测不能等同。条件配对按同GT Actor-view，列交集、各自缺失、并集、全候选负担；多视角重复PETR同一物体估计不当独立样本。Train无现成冻结eval文件标NotRun。历史Sealed已打开，仅回归。
6. `upright_3d_iou`是近似OBB体积交比；不是mesh交比。速度无可靠监督，NotEvaluated。统计字段over5_rate只在以米表示定位误差时用于严重定位解释；通用summary在角度/IoU中相同字段仅数学阈值，不能当严重定位率。

## 执行与复现

环境固定 `/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/env`，未重新安装。数据/原始输出位于 `/root/gpufree-data/hs-drive-experiments/m8b-petr-zero-shot`。权重原SHA `16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad`；numeric NPZ SHA `cf33dccb1d549c425a89cf346d6f05401c49f6cccbd5a579cb7d8937b5a6d591`。官方源固定 `f7525f93467a33707ef401c587a52d5e7b34de74`，入口[官方PETR](https://github.com/megvii-research/PETR/tree/f7525f93467a33707ef401c587a52d5e7b34de74)，实际复核使用本机固定源码。

全流程在全新独立输出目录/checkout执行；工具以`open('x')`保护证据，**现有目录原地重跑会拒绝覆盖**。复现时复制protocol并将root改为新的版本目录，保留原protocol SHA。首次导出使用系统python（读取旧数据工具），其余使用已有隔离环境：

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
python3 -m tools.m8b_prepare
python3 -m tools.m8b_run
python3 - <<'PY'
from tools.m8b_common import RUNTIME, run
for module in ['m8b_evaluate','m8b_pair','m8b_runtime_report','m8b_support_audit','m8b_figures']:
    run(module, [str(RUNTIME/'env/bin/python'), '-m', 'tools.'+module], timeout=1200)
run('tests', [str(RUNTIME/'env/bin/python'),'-m','unittest','discover','-s','tests','-p','test_m8b*.py','-v'])
PY
.venv-gpu/bin/python -m pytest tests/test_m8a_p1.py tests/test_m8a_assets.py tests/test_m8a_proxy.py tests/test_m7a_shadow.py tests/test_m5o_benchmark.py -q
```

现有证据只读检查不重跑GPU：

```bash
python3 - <<'PY'
import json, hashlib
from pathlib import Path
for a in json.loads(Path('reports/m8b_release_manifest.json').read_text())['files']:
    assert hashlib.sha256(Path(a['path']).read_bytes()).hexdigest() == a['sha256'], a['path']
print('M8-B artifacts verified')
PY
```

实际运行日志含命令、源码SHA、返回码、运行时间。保留失败：主环境无Shapely；改用已存在隔离环境；unittest发现路径冲突改discover；Evaluator list/ndarray算术修复；配对报告sources字段重名修复。全部发生在评价/报告，无权重/规则搜索或阈值改选。18项新测试、42项历史接口回归通过。诊断图仅用于展示，top10图框过滤不影响300候选原始结果。

## 时延范围与输出

每帧计读取+SHA+PNG、BGR、官方resize/crop+标定、normalize/pad、组装、H2D、完整网络、decode、CPU、最小JSON序列化；CUDA同步。E2E包含合同/有限性检查；不含模型初始化、输出文件磁盘写、Evaluator。首帧单列，余191帧冷暖文件缓存未控制。官方完整入口与分段入口首帧输出逐位一致。记录allocator峰值而非整个设备/进程GPU上下文；未做共享GPU压力、WCET证明或闭环时效认证。

输出仅DetectedObjectHypothesis，含原frame/timestamp、sim有效期timestamp+0.35s、独立墙钟完成时间、输入/模型/标定SHA、全部box/score/class及不支持字段。旧时间戳不更新为“新鲜”，离线结果不可向控制发送。位置/split关联仅在Evaluator sidecar中；模型metadata不含这些字段。
