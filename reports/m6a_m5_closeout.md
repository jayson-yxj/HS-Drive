# M5 Research Closeout

正式结题状态：**ResearchCompleted / CandidateNoGo / SafetyNotValidated**。

冻结基线 `349941e61abf706e462dce2f428149f9881be5ad`。本记录关闭当前研究扩展，不关闭感知或驾驶质量门禁；不创建 M5-P，不追加冻结 MobileNetV3 + DirectCenterHead 的训练、Loss 或蒸馏搜索。M6-A 没有训练新模型。

## 可复用的工程能力

- 六摄 RGB/Depth/Instance 同帧采集、CARLA 0.9.15 标定、Optical→HS Ego 右手坐标和原始时间戳。GT 传感器仅用于监督。
- ResNet Depth/Context、48 LID、原始 HeightSplat、四层 `[4,160,120]` BEV。保留原默认与所有候选。
- Spatial Evidence V2：surface 与 free-ray 证据非互斥，Unknown 独立 Mask；不是全体素可通行保证。
- MobileNetV3 前景研究分支、Actor Box Center 定义、全候选负担与唯一关联评价。
- M5-O Train/Validation/Calibration/Sealed 的位置分组和完整 3D 定位 Benchmark。已打开 Test 只作历史回归。
- Oracle 独立 Planner Worker、唯一 Tick Owner、5Hz/20Hz、双时钟过期保护和六秒 Safety；视觉始终 Offline Shadow。
- 5,275 个历史跟踪文件冻结、50,867 个历史资产 SHA 全量实核一致。原模型、原始数据、配置、失败日志和源文件均保留；见 [完整性报告](m6a_historical_integrity.json)、原 [M5-O 资产清单](m5o_artifact_manifest.json)。大型资产留服务器，Git 不能替代数据备份。

## 架构与结论索引

|阶段|实际研究与可复用产物|保留的失败证据/边界|
|---|---|---|
|[M4](m4_depth_distribution_analysis.md)|ResNet18/FPN、Depth/Context、Drivable、三冻结模型分布诊断|Road Point 60–80m MAE31.465m、Bias−31.453m；Mode不修复，概率偏近且分散。BEV内未稳定优于像素先验；近场/地图泛化退化|
|[M4.5](m45_summary.md)|Splat置信度诊断、三态Observed标签、Temporal Warp、Oracle驾驶|均值归一化抹去小概率幅度；自车粗盒过滤严重；Warp几何不是动态/遮挡质量|
|[M4.6](m46_summary.md)|自车实例过滤、时间参数化轨迹、停止目标一致性|104个表面/Free冲突；实际减速度/Jerk仍超限|
|[M4.7](m47_summary.md)|104格逐射线审计、碰撞几何参考、46组动力学|射线穿过只证明局部自由证据；碰撞盒不等于Mesh；小Brake低速突变真实存在|
|[M4.8](m48_summary.md)|非互斥V2、两帧Occupancy Pilot、独立控制候选|两帧可拟合不代表泛化；Town02碰撞表面召回低，物理占据仍未知|
|[M5-A](m5a_summary.md)|108帧/12位置，RGB与位置先验，异步Worker|条件AP不等于全空间定位；只有3个测试位置，阈值/局部变化不稳|
|[M5-B](m5b_summary.md)|Context及成对变化训练、分解图像/几何因素|跨位置变化有增益但精确空间P/R不足；Oracle深度干预非可训练上界|
|[M5-C](m5c_summary.md)|独立Obstacle Head、采样/量化诊断|可靠表面IoU高但障碍消失仍高分；Unknown无负监督不能变成Free|
|[M5-D](m5d_summary.md)|Presence/Absence点见证监督、真实P8候选|近场有收益，远场有效负监督不足；定位误差远大于Bin量化|
|[M5-E](m5e_summary.md)|中远场可见性、深度空间概率、独立运行时诊断|碰撞和渲染表面不同源几何并不完全一致；远场负监督仍不足|
|[M5-F](m5f_summary.md)|二维目标监督、RGB→Depth→BEV因子对照|二维与三维均不可靠；二值语义不能推出实例身份/完整Occupancy|
|[M5-G](m5g_summary.md)|官方COCO二维参考、FPN/高层对照、Depth研究|外部二维参考更强但非公平架构消融；Depth候选近场保护失败|
|[M5-H](m5h_summary.md)|有限视觉蒸馏、Replay及KL保护研究|两个seed增益不一致，Teacher优势不能自动迁移|
|[M5-I](m5i_summary.md)|可靠正负迁移、ImageNet Mobile对照|Mobile二维有收益；P8对远场小目标采样不足；Replay背景Bias问题|
|[M5-J](m5j_summary.md)|独立Mobile RGB分支、冻结深度与BEV聚合对照|原始BEV精确表面Precision约12.5%–17.4%；聚合升级NoGo|
|[M5-K](m5k_summary.md)|目标深度/自适应采样|Vehicle以外严格远场射线不足，按门禁未训练；采样增召回伴随背景错误超限|
|[M5-L](m5l_summary.md)|有限Vehicle残差、条件目标几何|残差推近背景，保护失败；2.5D固定地面依赖未验证接触|
|[M5-M](m5m_summary.md)|选择性接触假设、RGB门控、Shadow|接触Gold缺失；置信度牺牲过多覆盖；共享资源仍有长尾|
|[M5-N](m5n_summary.md)|Actor Box Center直接监督、Direct Head与Geometry Prior|训练每seed仅约64唯一候选；训练Loss低而位置泛化/风险失败|
|[M5-O](m5o_summary.md)|38位置、1223配置、独立风险Calibration、数据量曲线|数据增加改善均值，但Direct仍不能稳定超越Prior；尾部及风险门禁失败|

不同阶段改变过数据、Mask、选择指标、训练预算，以上不是跨阶段单变量消融，也不声称找到唯一根因。

## 最终停止依据

M5-O Train20/Val6/Calibration6/Test6位置；Train唯一候选642/658。从5到20训练位置，Direct Validation平均XY由4.95→3.61m、4.31→3.66m，说明更多独立样本确实有用；但两seed的同目标Geometry Prior仍更好。

历史封存集：Direct XY P95 **12.33/10.66m**、>5m比例 **35.4%/32.8%**；Geometry Prior P95 **8.69/8.28m**。候选/关联使23.7%/20.9%的已知目标无法可靠进入定位；已匹配定位也有严重错误。因此既不能归咎全是候选，也不能继续假设仅缺训练步数。

独立Calibration未找到满足Coverage≥30%、严重误差≤5%等联合门禁的阈值；最终准入拒绝全部。风险排序有相关性，但历史Test事后最佳覆盖仅17.9%/22.6%，不得据此调阈值。Geometry Prior本身也不合格，不自动升级它。

## 继续开放的门禁

可靠3D障碍感知、正式Occupancy、Temporal Memory动态/遮挡质量、Sparse Query实际质量、RGB-only Planner闭环、纯视觉安全、共享GPU严格20Hz全部保持开放。工程接口Passed、有限研究执行完成、泛化质量和安全许可是独立状态。

后续主线转为：独立Oracle仿真工程；一个有退出条件的成熟3D框架研究。不得自动再造第三个自定义3D Head，亦不把研究结题写成 M5Passed。
