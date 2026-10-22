# M6-A architecture decisions

基线 `feat/m5o-perception-benchmark-reset@349941e`；研发分支 `feat/m6a-system-baseline`。保持初始 Hybrid Spatial Driving 总体模块方向，不重设计主架构。

## A：M5研究结题

**Go：ResearchCompleted / CandidateNoGo / SafetyNotValidated。** M4–M5-O工程、候选、失败证据与完整SHA保留；停止当前冻结Mobile+DirectCenterHead自动训练扩展。不是Occupancy、Query或视觉驾驶通过。详见 `reports/m6a_m5_closeout.md`。

## B：Oracle工程主线

**Go限定工程推进，整体Partial。** 实际Town03三路线和故障降级可复现，13次7286Tick、约268.26m，无已记录碰撞或压线，过期规划正常控制0。路缘接触没有独立参考，不能宣称零路缘。

道路查询仍用完整地图检查；不短缩六秒Safety、不改默认Controller/Physics、不把5Hz规划的P95<200ms写成20Hz规划。静态障碍无绕行能力、停车过早、舒适性失败，后续应分别开发受控绕行候选和泊车任务，保持原控制器作对照。红灯/路权、复杂交通、倒车泊车未测。

## C：成熟3D感知研究

只选择[官方原始PETR](https://github.com/megvii-research/PETR)，固定 `f7525f93467a33707ef401c587a52d5e7b34de74` 与r50 DCNv2 C5配置。Apache-2.0源码；[官方安装说明](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/install.md)依赖Torch1.9/MMCV1.4/MMDetection3D0.17.1。

**本次P0 NoGo_AssetAccess；架构能力Unknown。** 独立Python3.8.20已创建、匹配MMCV wheel下载成功；两个Google下载端点不可达，官方Baidu目录/提取码可用但下载返回errno2。没有拿到模型字节，按资产失败退出条件停止Torch大包下载。没有证明依赖无法兼容4090，也没有真实PETR速度、显存、CARLA成绩；这些是NotRun。

没有P0通过就不转换/训练CARLA PETR数据。只读审查M5-O Train/Val837帧、5022 Actor配置的同步与Box参考可用；这不是PETR adapter通过。后续若明确可得官方权重，可以单独批准再次P0；先验证固定版本官方前向，再做原始模型域诊断。不得因本轮下载失败跳到第三个自定义Head。

### 坐标适配审查（待执行）

HS Ego为X前/Y左/Z上，CARLA原生Y右；相机Optical坐标通过保存的K/T投影。必须使用已变换8个OBB顶点均值，不能用Actor原点代替Box Center。PETR内部学习gravity center，导出时减半高度成为bottom center；比较前应恢复中心。

[MMDetection3D v0.17.1 LiDAR boxes](https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/core/bbox/structures/lidar_box3d.py)采用旧Yaw约定。适配必须锁版本，用0/90/180度非正方形盒的角点重建验证尺寸次序、Yaw符号和中心；坡道Roll/Pitch无法被直立Yaw盒精确表达，应保存误差并标Unsupported。当前未执行，不提供未经验证的Yaw公式。

官方类包含pedestrian/traffic_cone及多个Vehicle类；M5-O二值前景和粗Vehicle标签不能直接当十类可靠分类。全局对象与Actor-view分母必须分开，未知静态物体的未匹配输出不自动是物理FP。新适配若训练，只允许预声明预算、开发组选择、独立新最终位置评价。

## 控制隔离

`datasets/m6a_interfaces.py`定义OracleSceneState（Privileged）、RGBSpatialEvidence、DetectedObjectHypothesis和Unknown；保留原始frame/time、version、coordinate及valid_until。视觉/Unknown到oracle_planner消费者一律抛错，Unknown从不等于Free。Schema是防误接接口，不是可信认证或完整payload验证器；现有Controller仍使用原API。

本轮视觉模型、Depth48LID、HeightSplat、Temporal和Sparse均未替换。运行日志明示Privileged，未来视觉仍仅Offline Shadow，任何安全关键输入替换都需要单独批准。
