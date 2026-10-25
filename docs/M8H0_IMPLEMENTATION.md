# M8-H0 实现、证据和复现

本阶段执行只读原生来源审核。基础提交 `410d60b3449e3de3070ed0050f96ca998963e0fa`，分支 `feat/m8h0-native-scenario-source-review`。没有生成 Actor、传感器、RGB、GT 数据集或执行模型前向。运行端口2100的自有CARLA实例，加载Town02后保持异步；默认Physics、控制和模型文件未改。

## 资产与合同

外部目录：`/root/gpufree-data/hs-drive-experiments/m8h0-native-source-review`。原始 `native_spawn_points_raw.json` 保存一次调用的全部101个车辆Transform；`navigation_returns.jsonl` 保存恰好60次真实返回、顺序、episode、frame/timestamp，逐次flush。导航返回可重放，服务器随机过程不声称跨环境可复现。不是保存一个客户端seed就足够。

车辆只评估 `vehicle.tesla.model3`，行人只评估 `walker.pedestrian.0002`；尺寸来自冻结M8-F历史记录，原始asset来源SHA保留。Blueprint库存在其他车辆不等于它们的尺寸和接触已审核。本轮无新尺寸测量Actor。

`configs/m8h0/protocol.json` 在任何原生查询前冻结地图、车型、200车辆几何查询上限、60导航次数、最多4模板、20–40m范围和排序；Town02实际只有101个推荐点，全部审核。没有使用M8-G位置生成器继续搜索。Cone/Barrier不纳入组合。

`tools/m8h0_freeze.py` 先验证M8-G封存104项及6,640历史文件，再复算五个足迹回归。使用原16562条索引，不复制后修改其历史结果。历史空白不当作新Final；所有候选在模板选择前写入精确目标足迹/保守包络距离。

`tools/m8h0_inventory.py` 保留原生XY、yaw、Z，逐候选15点查原地面/车道/静态OBB规则。使用最近类车道仅提供声明层和方向，不将位置投影到车道中心；接受仍要求原`project_to_road=False`、Any类型完整足迹通过。原生heading与声明车道差异超原5度则附加拒绝。没有改0.10m inset或绕过NONE。

初始完整OBB与支撑间隙单独保存；仅当全部足迹有选定地面时构建竖直对齐的 **hypothetical** OBB。它不是落稳测量、不是批准的spawn命令，更不是靠关闭Physics固定位置。行人导航位置是地面目的地，若作为Actor中心会埋入地面；未来需按胶囊原点单独验证安全生成高度和稳定姿态。

`tools/m8h0_validator_replay.py` 用存储ray/lane回答离线调用原M8-F函数，确认相同输入规则下无新增接受。M8-F有Covered时不继续静态Box检查，新审核仍记录hypothetical OBB竞争来源，因此多原因集合可增加但不放行。初次过严的差分断言失败已保留，修正的是允许的诊断附加原因集合，无候选结果改动或新查询。

`datasets/m8h0_geometry.py` 计算完整OBB XY包络与Z区间、六摄名义pinhole投影。`visibility_certified=false`固定保留。相机内外参与旧rig一致，未输入Depth/Actor作为视觉模型特征；本轮根本没有模型。

`tools/m8h0_templates.py` 仅在已存清单上做有限配对：每个合格Ego各取一个离30m最近的车辆和行人，排序后按50m最小中心间隔选最多四组；没有全排列优化或未通过后的补抽。50m用于Development布局区分，不是新独立split规则。模板在静态视线OBB审核前冻结，不因遮挡警告替换。Ego/车辆/行人每组3对完整足迹关系保存。视线中心线与保守OBB相交只记警告，不冒充Mesh遮挡真值。

`tools/m8h0_map_identity.py` 保存API XODR、Town02包、BuiltData、Nav/TM二进制和运行服务器二进制SHA；未假设同名历史地图同版本，也不声称所有引用Mesh逐个验证。后续模板需绑定这些来源及环境库存SHA。

## 本次执行命令

所有输出采用独占创建，拒绝覆盖。本次命令完整记录在 `reports/m8h0_logs/*.json`，包括执行时间、返回码、源码SHA。以下是本次可审查的运行顺序；已完成目录不可直接重跑采样脚本。新的重复实验必须建立独立输出命名空间并冻结新协议，导航结果也会形成新清单，不得静默替换本次60点。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
.venv/bin/python -m tools.m8h0_freeze
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
```

地图初始化在另一个终端执行（仅自有服务；不用于改动他人的仿真实例）：

```bash
.venv/bin/python - <<'PY'
import carla
from tools.m8h0_common import ROOT
c=carla.Client('127.0.0.1',2100)
c.set_timeout(60)
w=c.get_world()
if not w.get_map().name.endswith('Town02'):
    w=c.load_world('Town02')
with (ROOT/'Town02.xodr').open('x') as f:
    f.write(w.get_map().to_opendrive())
print(c.get_server_version(), w.id)
PY
.venv/bin/python -m tools.m8h0_inventory
.venv/bin/python -m tools.m8h0_templates
.venv/bin/python -m tools.m8h0_map_identity
.venv/bin/python -m tools.m8h0_validator_replay
.venv-gpu/bin/python -m tools.m8h0_figures
.venv/bin/python -m tools.m8h0_reports
```

无需启动CARLA即可重复的只读核验：

```bash
.venv/bin/python -m tools.m8h0_verify
.venv/bin/python -m unittest discover -s tests -p 'test_m8h0*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_m8g*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_m8f*.py' -v
```

图像通过matplotlib绘制实测坐标，不是RGB采集或生成式场景。原始库存、逐点全部ray/OBB、导航顺序、模型隔离AST单测、历史回归、终态0车辆/0行人/0传感器读回均封存。最终源码/资产SHA见 `m8h0_release_manifest.json`；外部资产需单独备份，Git manifest不等于原始资产云端备份。

## 下一项申请的边界

只建议另行批准模板00、02两处真实Development Smoke，候选阶段保留其车辆路灯OBB视线警告。必须先核对地图/版本/原生XY、真实自然落稳、行人胶囊原点、最终车身支撑，再检查同步图像/身份/多视角/反事实及非法反例。Shifted pose本轮没有分配，需要下一协议单独预声明合法来源，不能恢复无限位置搜索。失败保持记录，禁止静默改用模板01/03或新增随机点凑通过。
