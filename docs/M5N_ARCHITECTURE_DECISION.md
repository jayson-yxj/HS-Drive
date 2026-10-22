# M5-N 架构与退出决策

基础`29ad67a`，分支`feat/m5n-direct-actor-localization`。原Encoder、Depth、48 LID、HeightSplat、Mobile参数、Controller和Runtime均只读。Actor中心研究与M5-M接触Gold完全分开。

## 参考合同

CARLA BoundingBox的`location`是相对Actor的Box中心，`extent`为局部三轴半尺寸，局部`rotation`必须保留。官方依据：[CARLA 0.9.15 Python API](https://carla.readthedocs.io/en/0.9.15/python_api/#carlaboundingbox)。历史已保存`get_world_vertices(actor_transform)`的8个顶点；中心由顶点均值得到，与Box局部旋转不冲突。HS世界/自车均使用右手坐标，CARLA到HS转换为S=diag(1,-1,1,1)。

逆Actor变换恢复局部Box中心与三条边；验证正交性、8顶点重建和世界坐标闭合。保存世界CARLA/HS、Ego、Optical Camera中心，光轴Z、欧氏距离、XY距离、K投影及局部尺寸/旋转。Actor原点、可见表面均值、可见底部均为不同字段。中心不是渲染Mesh质心，更不是轮胎或脚部接触真值。

历史采集在同步tick后、下一tick前读取静态冻结Actor，传感器帧号/时间戳和标定一致；旧文件没有逐Actor Snapshot帧号，因此不冒充动态严格Snapshot证明。新封存采集明确记录Snapshot帧号/时间和bbox局部location/extent/rotation。配对参考直接恢复原保存Transform，避免二次物理沉降。M5-M两帧配对失败的单帧几何单独审计；原像素sidecar依赖配对自车Mask，未自动复用，也未加入训练。

## RGB候选与网络

冻结Mobile101/102二值前景，每相机阈值0.9、最小6像素/3像素高度的历史连通域。GT仅用于离线Hungarian匹配(IoU≥0.1)，原生实例关联≥3像素、纯度≥0.9；合并或分裂候选不作唯一中心训练。模型没有可靠类别输出，`class_hint=foreground_unknown_class`。

576维冻结P8区域3×3特征加24维合法K/T/区域几何，40,643参数小MLP预测(logZ,du,dv)。偏移相对候选框宽高归一化；正Z范围1–80m，偏移限±2框。投影中心=框中心+偏移，不能把可见区域中心假装成Box中心。FP32头与几何，BF16冻结视觉特征。推理输入只接受六RGB与原白名单标定/元数据，无Actor/DepthGT/Instance/Map。

A固定M5-M Plane；B预声明区域原Depth概率均值的Expected和Mode，在同一候选像素位置摘要；C直接中心Head。额外geometry_prior仅用相同24维区域几何与前景置信度，仍依赖冻结RGB候选，不是完全无RGB的位置先验。它与C匹配样本顺序/训练步数，输入维数和参数量不同，不宣称等容量消融。

## 有限训练和资格

Train/Validation按完整位置组分离，复用K Train/Val和M开发对应划分，共各6组。旧K/L/M Test均只作回归。支持门禁Train≥40/Val≥30唯一候选、各≥4位置、至少2距离段；每类别不足20不作类别泛化主张。每种子每臂600步，AdamW lr0.001，batch32按位置均衡采样；Huber(logZ)+Huber(du,dv)。Val位置宏平均XY选择0/100/200/400/600步骤；非有限损失/梯度停止；预算不延长。

普通回归头不预测NLL方差，避免把方差当误差界。使用Validation实际XY误差拟合小型岭回归校准，输入仅预测范围、区域尺寸/完整性/前景概率；留一位置的分数用于Val拒绝诊断，最终全部Val拟合冻结到Sealed。90%经验倍率不是Conformal保证，且Checkpoint选择已用同一Val，不能认为完全独立校准。只有6位置，所有置信度解释必须附此限制。

阈值集合0/.2/.4/.6/.8，两个种子同时满足覆盖≥30%、定位≥20 Actor-view、≥3位置、P95≤5m、>5m比例≤5%，且同目标MAE/P95优于固定Plane和两个Dense摘要。Actor-view不是独立场景样本。无通过阈值时保留0作研究估计，不准入，停止当前训练路线。新的Test不重选阈值。

## 输出和退出

`DetectedWithEstimate`研究性Box中心位置；`DetectedNoLocation`保留高风险目标及可选假设，不能当成Free；观测不足使用Unknown；VerifiedContact本轮禁止。`localization_uncertainty`包含经验proxy误差、置信分数和经验90%半径，并显式`guarantee=False`。`research_qualified=False`体现Validation失败。

验证表明小样本可以拟合仿真中心，但RGB特征Head没有稳定优于几何先验，跨位置尾部与选择性覆盖未达标。停止本轮600步路线，不调整Loss/特征/预算去追Passed。不批准Offline Object-Center Query Pilot，保留参考合同和模型资产供未来独立协议研究。工程接口、可学习性、视觉增益、选择性可靠性、Runtime与驾驶安全是独立状态。
