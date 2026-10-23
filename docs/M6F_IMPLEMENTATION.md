# M6-F implementation and reproduction

独立研究分支`feat/m6f-vehicle-dynamics-observability`，基于`b0ee249`。主环境和默认控制源码只读。所有车辆数据为Privileged，禁止视觉控制。运行前确保没有其他CARLA tick owner、pytest、模型训练或其他GPU任务。

## 隔离环境

- 0915：原`.venv`和原CARLA目录，Guard时专用端口2100。
- 0916：`.venv-m6f-0916`，`/root/gpufree-data/simulators/m6f_CARLA_0.9.16`，端口2200。
- 服务端官方URL、HEAD、下载分段和SHA在`reports/m6f_execution_logs`。单文件下载速度慢，改为对同一资产分段续传，未另选版本。总获取预算900秒，SDK180秒；未下载附加地图。SDK镜像文件URL与PyPI官方同一SHA对应；安装补充opencv是旧Physics序列化模块的导入依赖。

```bash
.venv/bin/python -m venv .venv-m6f-0916
.venv-m6f-0916/bin/pip install carla==0.9.16 numpy==1.26.4 pyyaml pytest opencv-python-headless==4.10.0.84
# 在单独终端运行，UE禁止root；保留该进程直至批次结束。
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/m6f_CARLA_0.9.16 2200 Epic
```

## 实验冻结和执行

配置`configs/m6f/protocol.json`。固定50ms仿真、原子步配置。每次新加载地图、新spawn，无设置速度/瞬移制造动态正例。repeat_id只是重复编号，不虚称改变确定性PhysX随机种子。所有文件排他创建，已存在输出会拒绝覆盖；复现请在新checkout/空的m6f输出根执行，并保存原资产。

```bash
# 先完成所有代码编辑、测试并提交。批次运行时不要编辑已加载源码。
.venv-m6f-0916/bin/python -m tools.m6f_launch feasibility
# 人工检查真实omega/slip、帧对齐；telemetry_gate.json包含冻结的0916 Physics SHA。
.venv-m6f-0916/bin/python -m tools.m6f_launch responses
.venv-m6f-0916/bin/python -m tools.m6f_launch sham
.venv-m6f-0916/bin/python -m tools.m6f_launch overhead
```

`m6f_launch`捕获项目源码真实导入字节及编译code哈希。批前/末记录所有已加载外部模块的磁盘SHA和进程/GPU清单，明确它们不是内存镜像哈希。阶段变更必须等上一批退出，提交后新进程。原始JSONL逐帧落盘，结果文件只有完成才创建；异常保留failure.json。

每行记录请求、tick前后Applied Control、原始车身状态、真实Telemetry、RPC毫秒和墙钟时间。无tick重复RPC仅可行性批启用。所有Telemetry需外层frame bracketing；结构没有引擎子步时间。不要把debug torque改名brake torque，或将瞬时力与跨帧速度差强行对齐。

Sham：固定spawn后一帧后执行一次PhysicsControl；原P0、原参数重施加、历史P1四轮700→350 Nm；第二重复倒序。末尾恢复原Physics再销毁。P1仍是历史NoGo。

Guard在停止0916服务器后运行，防止GPU实例争用：

```bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m6f_launch guard
```

## 后处理与测试

```bash
.venv-gpu/bin/python -m tools.m6f_analyze
.venv/bin/python -m pytest -q tests/test_m6f_telemetry.py tests/test_m6e_guard.py tests/test_m6e_actuation.py
```

GPU仅用于CARLA渲染和停机后的离线画图；没有神经网络训练。实际完整测试、图表和最终报告步骤随交付记录。历史资产SHA在`m6f_execution_logs/base_freeze.json`，最终逐项核验。未执行前置条件不满足的工作包必须NotRun。
