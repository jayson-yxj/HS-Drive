# 独立CARLA自采路线

保留早先J3请求的独立评审。后续Bench2Drive本地读取授权不改变此路线依赖，也不补发CARLA启动授权。本轮没有RPC、Tick、Actor、传感器或采集。

本机VERSION与Python distribution均0.9.15；H0保存的9个Town02地图/导航/TM/Shipping文件重新校验一致。官方版本代码d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18，ContentVersions指向20231108_c5101a5。选定文件一致不认证所有引用Mesh或当前World身份。

[版本许可说明](https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/README.md#licenses)区分CARLA代码MIT、自有资产CC-BY、Unreal独立条款；README未给CC-BY精确版本。安装LICENSE的MIT不能覆盖所有资产。RSS/SUMO/Chrono/FBX等另有条件，本最小路线不新增这些集成，不用B2D路线/代码/标注/数据或Think2Drive权重。先为库存地图/车型/行人建立来源、署名和第三方清单，核实安装来源及适用引擎条款；不自动批准公开数据或商业用途。

现有HS采集/几何的静态导入闭包25个本地文件，未发现B2D/leaderboard/srunner/think2drive导入。AST不证明原创性或全部动态依赖；保留SHA，最小工具只复用HS接口及CARLA API，不复制B2D采集器。

| 数据 | 可复用接口与最小实现 | 必须验证 |
|---|---|---|
| 六摄RGB | camera_rig、FrameSynchronizer、Recorder，独立低预算配置 | 单Tick Owner；20Hz world/显式10Hz留帧相位；原始frame/time一致，漏帧不补邻帧 |
| K/T | HS配置+真实sensor transform，FP32/64几何 | 相机顺序、图像变换、标定SHA；运动中变换闭合 |
| 时间/状态 | world snapshot与每路image.frame/timestamp | 同一快照；规划量与实际控制/速度分开，不以后来查询代替 |
| Depth/BEV | raw BGRA、reference geometry、四层BEV | 光学Z/距离实证；不先转uint8；SurfaceCount非Occupancy；V2独立mask，Unknown非Free |
| 语义/实例 | 独立GT sidecar与I1隔离合同 | 身份关联、遮挡、静态背景、隐藏Actor单独审核 |
| 轨迹 | 实际未来ego位姿相对当前ego | 原始time/frame；末尾/漏帧mask，不跨episode；TM示范非最优专家，不自动用于模仿训练 |

现在可实施的范围是**离线合同工程**：以伪Client和合成运动测试身份先落盘、24路帧相位、漏帧拒绝、GT隔离、未来轨迹锚定及事务落盘。legacy session在地图拒绝前未保存完整身份，不能原样推荐新launcher。不恢复专用放置器或阈值射线补丁。

真实自采仍未就绪：H1自车支撑证据失败、I1版本RPC成功后地图拒绝、I1a连续31次版本RPC超时分别保留。I1b仍为StartupRootCauseUnresolved/NoRunMatchedUELog，单次授权已消耗。没有可核验新环境变化时，不申请盲重启。

未来前置条件闭合后，最多申请一次库存地图Development：1ego+最多2交通车辆、先无行人，20s/200留帧、180s墙钟、4GiB数据/64MiB日志。默认Physics，原生spawn自然落地，自定义路线非B2D XML，TM同步且单Tick Owner。身份/版本/XODR先落盘再校验，异常非零退出，不换图重试。此为建议预算，不是本轮授权。

验收建议：200帧所有通道原始frame一致、timestamp差≤1e-5s、几何闭合≤2e-3、深度平面既有门禁通过、无GT泄漏/损坏引用、完整SHA与清理。自然支撑、渲染身份各自验收，不能用零碰撞代替。首片仅Development；历史足迹/独立分组后才可扩展，不能宣称训练或Final资格。
