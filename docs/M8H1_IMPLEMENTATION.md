# M8-H1 实现、执行与早停

本次结果 `DevelopmentSmokeNoGo`。两处实际Ego在默认Physics下稳定，但原15点支撑查询无命中，停止下游。目标车辆/Walker/传感器/四阶段身份均NotRun。不能用可执行的采集代码存在，声称未走到的路径已经验证。

基础提交 `5e4701c0cdb81792d696c37157cf23d544843fb0`，分支 `feat/m8h1-native-template-actor-smoke`。外部证据 `/root/gpufree-data/hs-drive-experiments/m8h1-native-template-smoke`，所有写入独占创建。旧报告、数据、权重和默认控制不改。

## 合同与模块

- `configs/m8h1/protocol.json`：仅模板00/02、640×384/FOV100、dt0.05、单个+3mShifted、200Tick上限、原始稳定窗口及早停条件，在Actor生成前冻结。
- `tools/m8h1_freeze.py`：复核H0所有封存资产、地图/服务器文件及Rig SHA，复制两个批准模板作为可追溯只读来源。
- `tools/m8h1_preflight.py`：检查运行CARLA0.9.15、API XODR和静态库存。车辆/行人每个只有一个沿声明车道next3m候选，保留横向偏移；通过原M8F验证器后保存完整姿态、来源、OBB关系和历史足迹。无新随机导航。
- `datasets/m8h1_settling.py`：用原始速度/姿态判定，不平滑。至少40Tick、末20Tick；速度0.02m/s、窗口位置0.01m、窗口角度0.10°。XY相对原生Spawn漂移限0.02m、最终角差0.5°。允许并明确记录自然竖直下落。
- `tools/m8h1_actor_smoke.py`：按Ego→Vehicle→Walker顺序，每一步未通过就停止该模板。车辆固定brake1/hand_brake、无Planner或自动驾驶；不调用set_simulate_physics或apply_physics_control。保存初始Spawn RPC、首Tick后snapshot、完整轨迹、原始速度加速度及实际Box。
- `tools/m8h1_capture.py`：依赖所有实际前置门禁通过才导入/执行；本次未执行。包含现有RGB白名单包、Evaluator隔离、实际Ego绑定相机及24路同步收集。此路径不能标为端到端验证通过。预算仅8个保存关键帧，暖机与等待期间临时流不作为额外数据。
- `tools/m8h1_controls.py`：实际CARLA负例与历史回放/合成多层测试分开标记。真实负例不生成Actor。
- `tools/m8h1_diagnose.py`：只读解释当前射线起点、先前地面及Box差异，不追加射线实验或阈值修改。

## 自然下落与静态摆放合同

M8F `actual_check` 的“请求位置误差”原本用于静态放置。自然原生Spawn从+0.5m下落，不能把这个Z位移冒充执行器漂移。本轮保留原函数完整输出，再以实际记录的settled pose为支撑查询锚点，同时独立执行原0.02m XY/0.5°旋转约束。所有支撑语义、射线起点、地面间隙、车道和静态OBB规则保持不变。第二结果不是隐藏原始失败，也不会凭重设锚点通过：本次两套结果均无完整支撑证据。

实际OBB底部约0.21364m，原射线起点约0.20864m，先前同XY道路表面0.22m。向下查询全部NoRayHit。先前命中仅用于解释，未替代新支撑。不能把NoRayHit直接称悬空，也不能因为间隙在容差内就放行Unknown。

额外本地Box一致性诊断采用1e-5数值差异检测，发现历史反算local矩阵X平移与实际差0.156mm、extent相同。它是新加的来源差异标记，**不是历史通过阈值**。历史反算精度可能解释差异，但未作因果证明。无需解决该附加差异也已经因原NoRayHit停止。未调整任何数值后重新尝试。

## 测量错误与重测披露

首次立即读取Spawn返回Actor的get_transform获得全零缓存，首Tick后才出现真实位置，导致异常漂移误报。两份1Tick无效记录完整保存在 `invalid_measurement_attempt`，含原代码SHA。修正为以冻结spawn RPC作为初始参考、所有实际测量取tick snapshot；同位置/同车型/同约束重测一次，无新增目标或随机位置。总Spawn4次，其中2次无效仪器测量、2次有效Ego实验。真实支撑门禁失败后没有再次重试。

官方API说明Actor状态获取基于最近Tick，并提供frame一致的WorldSnapshot；因此Spawn命令成功与客户端已有有效状态是两项检查。[CARLA0.9.15 API](https://carla.readthedocs.io/en/0.9.15/python_api/)

## 实际命令与只读复验

下面为实际执行入口；当前输出已封存，重新运行生成脚本会因文件已存在失败。重现需要单独授权的新运行目录/命名空间，不能删除本次失败记录重新采集。本阶段已止损，不自动运行下一Smoke。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
.venv/bin/python -m tools.m8h1_freeze
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
```

另终端执行（仅本次专有服务器）：

```bash
.venv/bin/python -m tools.m8h1_preflight
.venv/bin/python -m tools.m8h1_actor_smoke
.venv/bin/python -m tools.m8h1_controls
.venv/bin/python -m tools.m8h1_diagnose
.venv/bin/python -m tools.m8h1_reports
.venv-gpu/bin/python -m tools.m8h1_figures
```

不启动CARLA的只读验证：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_m8h1*.py' -v
.venv/bin/python -m unittest tests.test_m8f_quality tests.test_m8f_evidence tests.test_m8g_geometry tests.test_m8g_evidence tests.test_m8h0_geometry tests.test_m8h0_evidence tests.test_m8e_source_time -v
python3 -m tools.m8h1_verify
```

实际19新增+73历史/时间戳测试；5项新CARLA负例、3项历史冻结回放、1项合成多层控制分别记录。时间戳单测不是本轮真实传感器同步。两个PNG为真实物理曲线和射线高度诊断，无伪造图像。所有程序命令/returncode/耗时/源码SHA在日志JSON，最终manifest封存资产。

## 清理及后续

CarlaSession按异常/正常出口销毁本轮Actor，恢复原World同步及天气；结束读回车辆/行人/传感器均零，异步恢复，然后仅停止本次CARLA PID。无Physics实现改动。

不再为这套放置器增加射线补丁或搜索位置。下一项只建议独立评审已有交通场景、ScenarioRunner或成熟标注数据，不能把本轮代码收束当成训练许可。

单位注记：原始轨迹键 `angular_velocity_deg_s` 沿用了Actor命名习惯，但采样来自ActorSnapshot；官方0.9.15文档对二者单位描述不一致。本轮该向量均为零且未用于门禁，保留原字段作为原始API数值，不据此作角速度单位或动力学结论。姿态窗口来自明确以度表示的Rotation。
