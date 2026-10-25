# M8-G 实现与复现

本阶段是有限几何诊断，结论为 `PlacementFeasibilityNoGo`。80 个候选完成后没有满足两处车辆+行人的前置条件，所以 RGB Smoke `NotRun`，不提供绕过门禁的采集入口。不会自动扩大搜索或训练。

## 环境与入口

基础提交 `8cde984ee606cc546c9d6c83962a7aaa95a08af8`；CARLA 0.9.15、Town02、端口 2100、Epic，沿用原 Physics。`.venv` 负责 CARLA/NumPy；`.venv-gpu` 只生成 matplotlib/Pillow 图像。独立 PETR 环境仅执行历史数据合同单测，没有模型加载/前向或训练。真实服务器信息在 `reports/m8g_logs/` 和外部库存记录中。

`tools/m8g_common.py` 的 `ROOT` 是本次不可覆盖的外部资产目录。所有写入使用 `open('x')`。下列命令是本次实际入口；在已完成目录重跑会拒绝覆盖。需要重新实验时，复制独立 checkout，并为 ROOT 和所有 `m8g_` 输出指定新的运行命名空间，记录新配置 SHA；不能删除当前证据后重跑。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
python3 -m tools.m8f_verify
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
```

另开终端，先确认当前不是其他工作的 CARLA 实例。下列命令是本次地图初始化的可复现形式；已有文件会拒绝覆盖。重新实验前必须先配置独立输出命名空间。

```bash
.venv/bin/python - <<'PYCODE'
import carla
from tools.m8g_common import ROOT
c = carla.Client('127.0.0.1', 2100)
c.set_timeout(60)
w = c.get_world()
if not w.get_map().name.endswith('Town02'):
    w = c.load_world('Town02')
ROOT.mkdir(parents=True, exist_ok=True)
with (ROOT / 'Town02.xodr').open('x') as f:
    f.write(w.get_map().to_opendrive())
print(c.get_server_version(), w.id, w.get_snapshot().frame)
PYCODE
```

本次已保存真实地图和 SHA。以下依赖顺序固定，实际运行的命令、耗时和 returncode 在同名 JSON 日志中：

```bash
.venv/bin/python -m tools.m8g_history
.venv/bin/python -m tools.m8g_diagnose
.venv/bin/python -m tools.m8g_scan
.venv/bin/python -m tools.m8g_controls
.venv/bin/python -m tools.m8g_source_followup
.venv/bin/python -m tools.m8g_scan_diagnostics
.venv-gpu/bin/python -m tools.m8g_figures
.venv/bin/python -m tools.m8g_reports
.venv/bin/python -m unittest discover -s tests -p 'test_m8g*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_m8e*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_m8f*.py' -v
```

历史 M8-B/D 单测使用 `tools.m8b_common.environment()` 的隔离环境及 `unittest discover -s tests -p 'test_m8b*.py'` / `test_m8d*.py`，不导入或执行检测模型。初次按模块名调用受到环境中同名 tests 包遮蔽，失败日志保留，改为 discover 后通过。

## 工程模块

- `datasets/m8g_geometry.py`：独立计算 OpenDRIVE line/arc、lane width/offset/elevation；支持两侧车道与 CARLA Y 翻转。未知 primitive/superelevation 拒绝证明。纵向道路端点残差也要受限，不能仅检查横向宽度。声明车道中心与 SDK 必须闭合；它是同源 OpenDRIVE 独立算术，不是独立 Mesh 真值。
- `tools/m8g_history.py`：先核验旧索引及 6,190 原始帧来源 SHA，统一 16,562 条记录。已恢复 polygon 输出 XY 精确距离；未知几何的 circle 输出保守下界。地图版本不完整、历史道路片区缺口和重复布局保持显式 Unknown，不形成 Final。
- `tools/m8g_diagnose.py`：冻结 8 请求、16 phase、240 射线；完整 A 重放。计划 OBB 被原验证器拒绝时，仅用已存在的全部 selected ground 重建 hypothetical box，绝不作为准许放置或无地面 fallback。分析每个命中、15 足迹点（含四角）、竞争环境/Actor 来源与历史六摄 Depth/独立语义/Instance。
- `tools/m8g_source_followup.py`：仅对同一批冻结点增加只读 `project_point`、Traffic Actor 包围盒及 Driving 类型查询。没有增加候选搜索。保留官方源码引用与可取得的本地 SHA。网络超时记录保留，不关闭 TLS。
- `tools/m8g_scan.py`：两处旧 Development 道路，每类每处固定十个沿车道距离，共 80 个；Present/Shifted 均验证。完整目标足迹历史预检在组合选择前执行。施工设施使用显式 lane-center work-zone 场景；场景语义改动与阈值不变分开报告。每个结果、拒绝、历史关系及组合冲突都保存。
- `tools/m8g_scan_diagnostics.py`：对全部 80 个已保存请求做离线空间注释，不增加候选。
- `tools/m8g_controls.py`：真实只读负例（全车越界、建筑、无命中、水面）与正向代理。Town05 高架路面仅是层面正例，不代表历史悬空目标被修复。

## 三类空间证据不可混淆

1. `cast_ray` 的 overlap 通道命中说明查询几何存在，不一定是可见/可碰撞的实体网格。交通牌 API 包围盒可能是触发体。21 点与触发体顶面匹配是强支持假设，但 API 没有返回 hit component ID，所以仍不绕过 NONE。
2. 相机通道 `project_point` 和 Depth/语义/Instance 提供可见表面交叉参考；看不到表面不能证明整个体积为空。
3. EnvironmentObject OBB 是保守近似。多来源竞争、高树木箱体或可见点在人体上方，都不能证明完整净空。

来源：[Map.cpp](https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/LibCarla/source/carla/road/Map.cpp)、[RayTracer.cpp](https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Util/RayTracer.cpp)、[BoundingBoxCalculator.cpp](https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Util/BoundingBoxCalculator.cpp)。官方源码版本说明实现机制；未声称已对运行二进制插桩。

## 验证与保留

`tests/test_m8g_geometry.py` 是合成数学反例；`tests/test_m8g_evidence.py` 检查真实记录完整性、预算、时序及不越权门禁。历史 M8-B/D/E/F 共 69 项回归另记。最终具体测试数和日志 SHA 见 `reports/m8g_test_results.json`。

初次索引生成重复 `sources` 参数、官方源码下载超时、历史 unittest 包解析失败均保留原日志，没有隐去失败。纵向端点 guard 是诊断算法补全，重新检查全部 240 点没有改变历史结论；旧源码版本保存在 ROOT/source_revisions。

所有新报告逐份带 kind、base、execution_commit、source SHA；最终 release manifest 封存原始资产、源码修订、配置、日志及图像。外部数据需连同 manifest 独立备份，Git 不保存巨量历史原始文件。服务器清理后只关闭本次启动的 CARLA 进程。
