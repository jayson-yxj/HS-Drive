# M5-M 架构决策

基线 `f2f4e1a`；独立研究分支；旧默认模型、控制器、48 LID bin、HeightSplat 不改。方法和门禁在 `configs/m5m_protocol.json` 预声明，Validation锁点后冻结源码/模型，再采集独立Sealed Test。M5-K/L测试全部属于历史回归。

## 输出合同

二值Mobile只能提供 `foreground_unknown_class`。`Detected` 是检出事件；`ConditionalPosition` 表示固定地面且可见轮廓底部就是接触点的条件假设；`VerifiedPosition` 必须另有可信接触/地面证据。当前正式 `status=Unknown`，另外保存 `conditional_status`、位置假设、原始时间戳、规则置信分数、敏感度和拒绝原因。Unknown候选不删除，不进入Free空间或控制器。

置信度是RGB连通域尺寸、实心度、底部前景概率及几何敏感性的固定函数，不是校准概率。敏感度不是位置误差上界。规则输入签名没有Actor/道路GT/深度GT；GT关联、匹配、可见性和接触状态只在离线评测路径。

## 接触质量与训练停止

Gold须同时具有：多个道路碰撞探针与渲染表面一致、可独立确认的可见轮胎/脚部接触、稳定姿态、无截断/遮挡/悬浮/穿透歧义。AABB底部、最低实例像素、骨骼原点、物理轮半径均不足以单独成为Gold。碰撞几何不等于渲染Mesh。

Contact Head要求Train/Val各至少50个Gold正样本、50个可靠不可定位负样本、4个独立位置、每目标类别至少20个。样本不足时不训练，不以Candidate降格替代Gold。新增开发场景用于确认接触证据是否可采，不以凑数量为目标。至多2种子预算不意味着必须训练。

当前固定平面规则只输出候选。Selective gate最低30%全可见覆盖、20匹配/3位置、P95≤5m且≤旧同分母基线、>5m错误率≤5%，两个种子都满足。没有合法Validation锁点时，阈值0仅保留诊断，不批准部署。

## 固定残差

`P_mix=(1-g)*P_original+g*P_residual`，沿深度维混合归一化概率，不能混合期望深度假装得到同一分布。RGB门控只用冻结Mobile二值概率，阈值集合0.5/0.9/0.95在Validation检查。Oracle Vehicle类别门控明确为诊断。每项近场、背景Bias、逐位置、空间概率、BEV P/R/FPR保护必须同时满足；目标子集改善不能覆盖全场失败。本轮不更新任何残差参数。

## 局部平面

固定平面不改。RGB候选用原Depth预测在RGB目标下方、前景概率<0.1区域拟合；没有可靠RGB道路分割，因此只能称局部平面候选，不能称可信道路法向。拟合失败/坡度过大拒绝。Oracle局部渲染Road平面只用于分别隔离底部像素误差和平面误差。没有训练新道路网络。

## Runtime

Worker无CARLA客户端，无tick/control。主进程唯一TickOwner，容量1队列最新完整六摄包，2Hz墙钟发送；输出使用原始仿真时间戳，250ms墙钟总年龄拒绝。两次Alone/Loaded-idle/Shadow/Recovery，第二次交换中间条件。冷推理、CUDA/模型加载、初始化期间tick独立报告；每阶段20预热+160测量tick。若P95超过max(1.5×baseline, baseline+20ms)或单tick>1000ms，停worker并恢复Alone。

`put_nowait`只保证调用不等待队列空位，不能证明后台序列化、内存拷贝或共享GPU对tick无影响。静态摄像机试验不等于驾驶周期认证，6秒Safety保持原样。资源隔离与消息隔离分开验收。

## 四个独立结论

工程接口、Gold接触质量、选择性定位质量、在线资源隔离不能互相替代。具有字段完整的对象合同不代表已有可信目标位置；本轮不会自动接入Object Query、SparseDrive、Occupancy、Temporal或安全规划。

## 最终归因与数据门禁说明

Validation残差远场退化的准确归因是未关联到记录Actor的渲染前景，不是可靠背景。35–80m共783可靠射线，其中575背景、208未关联前景、0可靠关联Actor；Binary Mobile把208前景全部门控后，Vehicle-only残差把它们推向错误距离。不能根据标签缺少ActorID推断它们是背景，也不能假定全部为Vehicle。总体/逐位置保护仍失败，研究停止结论不变。

新增封存两帧配对姿态超差，按预声明0.001容差剔除，保留未过滤索引、原始数据及失败日志。主封存评价34帧/4位置，仅表示合格配对子集；封存数据质量状态Partial。此门禁独立于RGB模型表现，未据测试选择阈值或新增训练。
