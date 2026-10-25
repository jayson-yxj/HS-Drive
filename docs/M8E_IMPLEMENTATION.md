# M8-E 实施与复现

本阶段源码仅新增 m8e/M8E 文件。执行基于提交037ddba；每次进程日志记录执行父提交、当时源码SHA、命令、退出码、耗时及输出SHA。最终Git提交封存 reports/m8e_release_manifest.json。原始资产需要保留服务器实验目录，单独克隆Git不包含全部数据。

## 实际环境与目录

CARLA0.9.15、RTX4090、Epic离屏，端口2100，固定仿真步长0.05s。原六摄640×384/FOV100及 configs/cameras.yaml 不变。24相机传感器同步输出六摄RGB、Depth、Instance及独立Semantic。

- .venv/bin/python：CARLA、OpenCV、NumPy，负责API与采集。
- .venv-gpu/bin/python：PIL、NumPy、Matplotlib，负责新数据审计/可视化/单测。
- /root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/env/bin/python：已有Shapely，负责历史定位统计。没有加载PETR或安装依赖。
- /root/gpufree-data/hs-drive-experiments/m8e-groundtruth/capture/GROUP/rgb：只含RGB和白名单packet。
- 同组 evaluator：特权数据、完整源对象库存和原始传感器记录。
- 实验目录 audit：完整逐帧ObjectReference、未链接像素区域和冲突；official保留可下载的官方源文件。

## 源合同与质量规则

Actor记录使用同一world snapshot的变换。新增 require_source_frame 在未来审计入口拒绝缺失snapshot、非有限或不同帧时间以及缺失模态。此收尾防护对本轮32帧全部通过，未改任何身份结果；执行时旧源码按SHA保留在logs。

EnvironmentObject静态清单记录查询帧和episode，不当作每帧新Mesh真值。匿名level box没有可连接实例ID。完整库存保存，但审计局部搜索只使用90m内、有限且最长边小于40m的源Box；巨大静态对象/未链接区域仍Unknown，不是完整场景标注。

深度为光学Z，由像素K反投影再转CARLA world，与完整OBB进行点包含检查。候选必须语义兼容、唯一来源，至少90%点在容差0.25m的OBB内，独立语义相机一致率至少95%，至少3个有效且裁剪后保留像素。裁剪资格沿用原增强332/2.2≈150.91原图行，即v>=151。多视角按不同相机计数。

受控prop的SDK semantic_tags实测为空。Development阶段冻结修正：只能凭明确受控资产身份、唯一几何与实测语义建立连接；不能把所有Dynamic像素映射为Cone/Barrier。源ID与渲染ID从未按数字相等匹配。46个接受Present目标的全图相同tag+编码在Empty/Removed中均消失；Shifted允许离开视野。

历史预测关联沿用冻结的一对一投影提议：score0.1、兼容类别、投影IoU>=0.1；支持视角须实例BBox IoU>=0.5、Mask Recall>=0.8、来源purity>=0.9及>=8保留像素，无竞争对象。中心误差不参与身份资格。历史没有独立Semantic相机/完整EnvironmentObject侧车，证据强度低于新采集；必须保留这一限制。

官方依据：[0.9.15 API](https://carla.readthedocs.io/en/0.9.15/python_api/)、[传感器说明](https://carla.readthedocs.io/en/0.9.15/ref_sensors/)、[官方Tagger源码](https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Game/Tagger.cpp)。GetActorLabelColor使用UE Actor.GetUniqueID低16位；不是Python Actor.id。保存的历史opaque编码为256*G+B，canonical UE low16为G+256*B；原BGRA保留。SDK实际枚举优先于旧网页上的语义类别示例。服务器两项raw下载失败记录仍在，不声称全部源文件已下载。

## 可直接执行的只读验证

在仓库根目录运行：

~~~bash
.venv-gpu/bin/python -m unittest discover -s tests -p 'test_m8e*.py' -v
unshare -n .venv-gpu/bin/python -m tools.m8e_input_isolation /root/gpufree-data/hs-drive-experiments/m8e-groundtruth/capture
python3 -m tools.m8e_verify
~~~

第一条为18个新测试。M8-D16和M8-B18的原始回归命令及环境记录在 reports/m8e_logs/m8d_regression.json 和 m8b_regression.json。输入probe只校验RGB接口，不做模型前向；3个读文件canary与32个GT注入应拒绝。网络namespace加Python open guard只防误用，不是恶意原生代码隔离。

## 采集与审计的复现次序

现有实验全部采用追加写入，重复运行会拒绝覆盖。重建采集必须使用独立目录/工作副本，在 tools/m8e_common.py 中为ROOT配置新实验目录，并保留原 protocols；不要删除历史文件来重跑。ID/frame/render编码可能随episode改变，复现的是合同和配置，不承诺字节一致。以下是实际命令形式，不要求重新自动采集：

~~~bash
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m8e_capture --groups m8e_development_00
.venv-gpu/bin/python -m tools.m8e_audit --groups m8e_development_00
~~~

Development_00先做smoke，修复SDK包装器身份和prop标签问题后冻结规则，再执行剩余Development以及Validation/Calibration/Held-out。所有实际组和命令见 configs/m8e/protocol.json 及 logs/capture_*.json、audit_*.json。几何分配器只在新协议不存在时运行：.venv/bin/python -m tools.m8e_allocate；现协议已冻结，不应重生成以换取更好结果。

离线完整顺序为 m8e_historical（使用既有PETR隔离环境和 tools.m8b_common.environment()）、m8e_report、m8e_supplement、m8e_figures、m8e_temporal_identity_check、m8e_closeout、m8e_finalize。均通过 python -m tools.MODULE 形式；实际命令/运行环境以logs为准。历史环境调用示例：

~~~python
from tools.m8e_common import run
from tools.m8b_common import environment
run('historical_new_reproduction',
    ['/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/env/bin/python',
     '-m', 'tools.m8e_historical'], environment=environment())
~~~

输出已存在时会失败，因此该示例只用于配置了独立输出根的重建副本。最终verify是只读、可重复执行的。

## 已知失败与证据层级

全部服务器真实：API查询、32帧静态渲染及768载荷校验。离线真实数据：OBB/实例审计、历史154背景候选重审、12个身份支持目标定位、21个远场四阶段图。合成单测：冲突身份、旋转Box、GT注入、缺失/过期源帧。人工：没有独立审核员，仅AssistantVisualReview。

保留初始缺PIL/缺Shapely运行、controlled包装器错误、空prop语义、相机重复组件计数修正、官方下载失败。完整失败账本见 m8e_failure_ledger.json。没有调低像素、语义或几何阈值。

未完成：真实路面接触、自然布局有效性、完整静态Mesh身份、新最终集独立性及足够远场支持。尤其生成器ground fallback没有记录逐目标命中来源，不可据此建立接触真值。新PETR诊断、训练和驾驶均NotRun。
