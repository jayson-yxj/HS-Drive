# M6-A implementation and reproduction

基线 `349941e61abf706e462dce2f428149f9881be5ad`，分支 `feat/m6a-system-baseline`；第一实现提交 `98ec296`。逐实验启动源码保存在各 `reports/m6a_*artifacts/source/manifest.json`，最终清单为 `reports/m6a_artifact_manifest.json`。原始文件和失败记录不覆盖。

## 环境与实际执行

CARLA0.9.15、RTX4090 24GB、驱动580.126.09。现有 `.venv`（CARLA）和 `.venv-gpu`（Torch2.8）不安装PETR依赖。新增代码仅以 `m6a_` 命名；未修改默认Planner、Safety、Controller、Physics、Encoder、Depth48LID或Splat。

13次CARLA运行、7286Tick；三正常路线、动态前车两种fixture、低速短路线两次、静态障碍、Emergency/滞后/崩溃/坏输入及最终schema smoke。完整JSONL含原始仿真时刻、控制墙钟、逐阶段耗时、命令、测量、原因和Actor快照。没有调用视觉推理控制车辆。

以下命令在项目根目录执行。运行产物已有，复现必须使用新的 `--output` 和 `--artifacts`。报告生成器也拒绝覆写；完整重建应在独立checkout和新的产物目录执行，不删除本轮报告。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
git switch feat/m6a-system-baseline
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_geometry_extended.py tests/test_imaging_rig.py tests/test_session.py tests/test_snapshot.py tests/test_m5n_sdk.py

# 独立终端；仅一个服务器与Tick Owner。无本轮共享视觉工作负载。
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic

# 另一个终端，正常路线
.venv/bin/python -m tools.m6a_oracle_drive \
  --output reports/m6a_repeat_routes.json --artifacts reports/m6a_repeat_routes_artifacts \
  --scenarios straight mild_curve junction_turn

# 故障、静态障碍与低速停车；不会实现绕行或倒车
.venv/bin/python -m tools.m6a_oracle_drive \
  --output reports/m6a_repeat_faults.json --artifacts reports/m6a_repeat_faults_artifacts \
  --scenarios emergency static_obstacle stalled_worker worker_crash invalid_input low_speed_stop

# 显式仿真特权前车：1.5m/s恒速至tick300，然后制动
.venv/bin/python -m tools.m6a_oracle_drive \
  --output reports/m6a_repeat_follow.json --artifacts reports/m6a_repeat_follow_artifacts \
  --scenarios dynamic_lead --lead-motion scripted
```

原始完整执行命令与失败日志归档 `reports/m6a_execution_logs/`；运行中间日志原件在 `runs/m6a_logs/`。`tools/m6a_report.py`读取所有本轮result，计算指标并绘制路线、未滤波Jerk和周期。新增重复文件会改变聚合分母，应单独报告，不能偷换本轮13次结论。

## 指标定义与合同

`diagnostics/m6a_oracle_metrics.py`：实际行驶距离为连续原始位置XY弦长之和；横向误差为HS world点到路线线段最短距离，原生CARLA world Y显式取负。路线完成率是路线弧长投影比例，不是实际里程；倒车/绕行等未来任务需独立定义进度。

正常路线通过需同时：完成≥95%、无碰撞/压线/当前footprint边界、CTE P95≤0.5m、最终速度≤0.08m/s、过期规划正常控制0。故障停车另看最终停稳及fault-to-brake≤0.4s；该门槛只限制本轮低速fixture，不是安全响应规范。跟车最终前车停住不应要求穿过前车完成全路线，但也未据此宣布完整导航成功。

控制周期为实际相邻apply时刻差；P50/P95/P99/Max和>50/>55ms分别保存。tick完成deadline miss、控制周期超50ms和Controller仿真时间超时是不同指标。Planner失败率分母为实际Worker输出，Safety拒绝和异常同时计入并列原因；无输出不伪造成功率。规划过期收到次数与降级控制Tick分开计数。

障碍间距是路线弧长减名义车身半长的代理，不是精确接触距离。Collision记录其他Actor类型和冲量，Lane传感器记录线型；无专门路缘接触参考，因此路缘状态Unknown。交通灯固定绿，不验收交通规则。Safety为原有近似footprint检查，非精确连续旋转Mesh碰撞。

`datasets/m6a_interfaces.py`仅建立明确来源边界。OracleSceneState允许HS Ego或显式HS world；视觉仅HS Ego OfflineShadow。原始时间戳不得续期重写，未来/过期/非有限值拒绝。Unknown不是Free；新类型不直接连接执行器。

## PETR隔离尝试与退出

官方源码固定 `f7525f93467a33707ef401c587a52d5e7b34de74`，目录 `/root/gpufree-data/hs-drive-experiments/m6a-petr/source`，另存 `source.bundle`。官方原始r50 C5配置、许可证和requirements均有SHA。不是PETRv2/StreamPETR。选择及预算在 `configs/m6a/protocol.json`，45分钟上限、官方权重下载失败即停止；没有训练预算。

实际命令（在新的实验根目录复现，避免覆盖现有环境）：

```bash
PETR_ROOT=/root/gpufree-data/hs-drive-experiments/m6a-petr-repeat
mkdir -p "$PETR_ROOT"
git clone https://github.com/megvii-research/PETR.git "$PETR_ROOT/source"
git -C "$PETR_ROOT/source" checkout f7525f93467a33707ef401c587a52d5e7b34de74
python3 -m venv "$PETR_ROOT/bootstrap"
"$PETR_ROOT/bootstrap/bin/pip" install uv gdown
"$PETR_ROOT/bootstrap/bin/uv" python install 3.8
"$PETR_ROOT/bootstrap/bin/uv" venv --python 3.8 --seed "$PETR_ROOT/env"
"$PETR_ROOT/env/bin/pip" download --no-deps --only-binary=:all: mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html -d "$PETR_ROOT/wheels"
"$PETR_ROOT/bootstrap/bin/gdown" 1c5rgTpHA98dFKmQ9BJN0zZbSuBFT8_Bt -O "$PETR_ROOT/petr_r50_c5.pth"
# 本次此处权重获取失败，不能继续声称P0或P1完成。
# 已尝试且因资产门禁中止的依赖命令：
"$PETR_ROOT/env/bin/pip" install 'torch==1.9.0+cu111' 'torchvision==0.10.0+cu111' \
  -f https://download.pytorch.org/whl/torch_stable.html
```

第一次Google Drive网络不可达，第二usercontent端点超时；官方Baidu提取码`petr`验证/目录列举成功，目标435,218,261字节，但sharedownload返回errno2。不得据此推断文件删除、许可证不允许、框架不兼容或4090显存不足。未获得权重SHA、模型导入/官方前向/峰值显存/推理延迟均NotRun。保留stdout、官方资产列表和环境日志；没有绕过访问控制。

`tools/m6a_feasibility.py`只读Train/Val837帧的原始字段审查，不创建PETR训练数据。GT位置/Depth/Instance不进入模型。任何未来适配先通过P0，再验证相机顺序/预处理K、旧Yaw/尺寸/中心转换、坡道直立盒误差；打开过的M5-O Test只能回归。

## 失败与修正

1. 首次批处理三个正常路线完成后，新增dynamic_lead误映射不存在的同名历史route，异常停止。改回straight别名，剩余单独运行；之前三条实际结果保留。
2. 首次低速短路线部分重叠全仓测试负载，负载字段在聚合报告更正；独立复测仍提前停住。不得选择较好一次覆盖失败。
3. Actuator前车实际只约0.24–0.29m/s，不宣称达到1.5m/s；另加显式constant-velocity特权fixture并报告，不与动力学控制等同。
4. 初版Oracle envelope默认标签为hs_ego，但实际状态与障碍一直是HS world。已修正元数据并实测Emergency smoke。旧原始记录保留，聚合报告逐条标注更正；控制数值、标定和离线Y转换无变化。
5. SDK测试两次使用不存在的文件名，原失败输出保留；随后正确集合18通过，GPU跳过项对应SDK集合15通过，不能相加声称独立总数。

测试全仓333通过/6跳过；6项因GPU环境无CARLA，另SDK集合15通过覆盖。GPU实际算术smoke正常但不是PETR推理。`pip freeze`前后仅editable HS-Drive的Git引用变化，依赖版本不变；历史源码和资产最终再核对。

验收入口：`reports/m6a_acceptance.json`；有限路线工程成功、感知研究停止、运行时Partial和纯视觉SafetyNotValidated分别保留。
