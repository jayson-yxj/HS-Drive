# M8-F 实施与复现

基线12bd124b443cc01c60ee4251fd1be20d868928c1；分支feat/m8f-grounded-scene-quality。原M8-E文件未修改。所有工具追加输出，避免覆盖旧实验；重复生成同名结果将失败。

## 工程接口

- configs/m8f/contract.json：采集前冻结全部容差及Smoke停止条件。
- configs/m8f/protocol.json：两个Development Ego、每类Present/Shifted车道请求，采集前冻结。
- datasets/m8f_quality.py：纯几何足迹距离、表面选择、合取资格规则。
- simulator/m8f_placement.py：CARLA射线、车道与环境Box请求检查，实际Actor二次检查。
- tools/m8f_capture.py：独立静态采集器；合法性先于实际目标spawn。RGB与特权Sidecar分目录。
- tools/m8f_audit.py：复用M8-E身份规则的本阶段版本，不改旧生成器或结果。
- tools/m8f_history.py、m8f_history_supplement.py：历史预索引及后补原始帧索引，分别存档。
- tools/m8f_region_review.py、m8f_certificates.py：评价区域Mask和所有ObjectReference独立资格。

固定规则：车/行人/施工设施限定Driving或Parking/Sidewalk/声明路边区域；5×3足迹采样；车道内缩0.10m；射线上8m下15m；候选层与Waypoint高度差不超过0.5m；不同合法层差0.5m即多层拒绝；支撑高度差不超过0.12m；最终底部间隙[-0.03,0.12]m。请求Z由真实选中命中与资产底部推导，加0.03m预声明间隙。无源命中绝不补固定Z。

最终核验记录请求/实际Transform、完整OBB、底角/足迹点、从底部下方射出的真实命中、支撑高度差及区域/静态重叠。底部起射避免命中目标本身；未验证精确轮胎/脚底Mesh，等级仅GroundedByGeometryProxy。朝向来自所选Waypoint；超出既定停车语义的朝向不支持。

身份沿用M8-E：≥3保留像素、深度1.5–80m、源OBB包含率≥0.9、容差0.25m、独立语义一致≥0.95、唯一来源；多个相机才是MultiView。不读取PETR输出决定样本保留。

## 实际环境

CARLA0.9.15，原Rig640×384/FOV100，固定步长0.05s；RTX4090仅用于Epic离屏渲染。8帧RGB采集全在Town02，Town04/05仅射线反例查询，无新增RGB。没有更改VehiclePhysicsControl，静态Actor冻结不当作动力学证据。

.venv/bin/python包含CARLA/NumPy2/OpenCV，用于仿真；.venv-gpu/bin/python用于PIL/NumPy/Matplotlib离线审计；旧PETR隔离Python只为M8-B/D CPU几何回归提供Shapely，未加载网络。没有安装或修改依赖。

## 可重复只读验证

仓库根目录执行：

~~~bash
.venv-gpu/bin/python -m unittest discover -s tests -p 'test_m8f*.py' -v
.venv-gpu/bin/python -m unittest discover -s tests -p 'test_m8e*.py' -v
unshare -n .venv-gpu/bin/python -m tools.m8e_input_isolation /root/gpufree-data/hs-drive-experiments/m8f-grounded-quality/capture
python3 -m tools.m8f_verify
~~~

M8-B18、M8-D16的原命令保存在 reports/m8f_logs/m8b_regression.json 和 m8d_regression.json；环境为 tools.m8b_common.environment()。本轮17、旧M8-E18及上述34合计69通过。

Input probe为8包/48RGB，不做模型前向。GT字段注入与Evaluator路径canary拒绝；Python audit hook不是防恶意native代码的沙箱。

## 实际执行命令及重建

以下采集命令只应在新的独立实验根中重建，当前ROOT和报告已封存，不能删除旧结果来重跑：

~~~bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv-gpu/bin/python -m tools.m8f_history
.venv/bin/python -m tools.m8f_allocate
.venv/bin/python -m tools.m8f_capture --groups m8f_development_00 m8f_development_01
.venv-gpu/bin/python -m tools.m8f_audit --groups m8f_development_00 m8f_development_01
.venv/bin/python -m tools.m8f_negative_controls
.venv/bin/python -m tools.m8f_water_probe
.venv-gpu/bin/python -m tools.m8f_counterfactual
.venv-gpu/bin/python -m tools.m8f_reports
.venv-gpu/bin/python -m tools.m8f_history_supplement
.venv-gpu/bin/python -m tools.m8f_region_review
.venv-gpu/bin/python -m tools.m8f_certificates
.venv-gpu/bin/python -m tools.m8f_figures
.venv-gpu/bin/python -m tools.m8f_geometry_figures
.venv-gpu/bin/python -m tools.m8f_closeout
~~~

m8f_common.ROOT指向/root/gpufree-data/hs-drive-experiments/m8f-grounded-quality。独立重建副本须改为新的ROOT和报告/配置目标；现版本为本轮冻结实验工具，不是通用批量数据服务。实际日志记录完整命令、环境适配、源码SHA与运行时长。追加写入防止覆盖；重建不会保证CARLA episode/frame/ActorID字节相同。

第一次采集在NumPy2 ndarray.ptp调用失败，尚无RGB；失败目录failed_preflight_development_00_numpy2保留。兼容修复后同协议重试，不调整阈值。一次closeout重复字段失败及已生成报告也保留。每次执行的源码版本和SHA见logs，最终release manifest封存全部资产。

## 输出及证据边界

原始capture/GROUP/evaluator保存所有射线、源清单、实际姿态和Depth/Instance/Semantic；rgb仅RGB与白名单packet。audit保存身份竞争/未链接区域。reviewed_regions的Mask仅表示实际来源支持像素，补集未知；certified_references为每个观察对象增加独立资格。history原索引与后补索引各自独立，不能静默覆盖preflight。

真实反例：越界、建筑区域、无命中，以及历史可疑XY的新Water查询。合成反例：两个合法地面层、水面语义、无命中、屋顶、错层、OBB距离及任一资格失败。真实多层道路闭合、足部Mesh接触、多Actor相互穿模、独立Final和全部背景类别仍未完成。

官方API依据：[CARLA0.9.15 Python API](https://carla.readthedocs.io/en/0.9.15/python_api/)。cast_ray返回带语义的命中列表，Waypoint/EnvironmentObject只能提供几何代理；是否实际匹配本场景仍由保存的原始返回与审计判断。

Smoke失败后没有新增后续Split或PETR调用。最终只交付可审计Partial，不宣称自然训练数据已就绪。
