# M4 感知质量诊断与 BEV 有效性验证

**结论：M4 质量门禁继续开放。模型确实使用视觉信息，但还没有稳定获得优于简单空间先验的 BEV 几何。应把主评价转向 BEV 内的可见表面、概率投影和道路泛化；不继续盲目延长深度训练，不立即开展大型 Occupancy/Temporal 联合训练。**

本轮只读取三个已有 checkpoint，无优化器、无反向更新、无新1200步训练。原报告和模型逐文件哈希核验未变。222帧固定验证集、867,467个有效采样点，三模型sample ID和连续GT/UV/valid mask哈希完全相同。原始RGB期望MAE重现误差最大1.62×10⁻⁷m。所有扰动、Oracle及重加权结果仅作诊断。

## 1. 实验控制与统计口径

固定：ResNet18、48个LID区间（1.5–80m）、六摄640×384输入、stride16采样、四高度层160×120 BEV、seed7、相同299训练样本、1200更新步、16帧/场景的训练期验证选步预算、原始222帧全验证集。未访问test进行本轮诊断。

| 配置差异 | Joint | Far balanced | Road point |
|---|---|---|---|
| 实际checkpoint步 | 1200 | 800 | 1000 |
| 深度损失 | CE + log-depth SmoothL1 | 按四距离段等权 | 按四距离段等权 |
| 深度log损失/Surface权重 | 1.0 / 0.2 | 1.0 / 0.2 | 1.0 / 0.2 |
| 道路训练标签 | v1 | v1 | v2路口overlay |
| 道路损失权重 | 0.2 | 0.2 | 0.5 |
| checkpoint选择 | 验证depth MAE | 验证四段macro AbsRel | 验证四段macro AbsRel |
| 道路头 | 1×1 | 1×1 | 1×1 |

Joint→Far同时改变损失聚合与选步；Far→Road同时改变道路标签、道路权重及最后选定步数。不存在严格单变量因果结论。道路诊断统一使用旧v1标签及有效掩码，因此本报告Road IoU=0.541，与上轮v2的0.564口径不同。

概率指标先在相同`depth_uv`对FP32 logits双线性采样，再Softmax。Mean=Σp·center；Mode=最大概率Bin中心（不是除以非均匀Bin宽度的最大密度）；Median=累计概率首次≥0.5的中心。MAE以连续GT计算；Signed Bias=预测−GT。熵=−Σp log p/log48。GT Bin Probability按GT所属区间取值。

±1/3/5m概率质量主要口径为窗口内Bin中心概率之和；另存`*_uniform_bin_assumption`，按每Bin内部均匀假设对窗口重叠积分，不能理解为网络提供了Bin内分辨率。Distribution Expected Absolute Error=Σp|center−GT|，与|Σp·center−GT|不同。统计为有效点加权；BEV空间TV、特征距离、Surface分组误差采用逐帧均值，分母均记录。

双峰比例是启发式：两个局部峰概率均≥0.05、间隔≥5m、次峰≥主峰25%、谷底≤较小峰50%。它只用于描述，不是唯一“多峰率”定义。全部分摄、距离、场景和场景×摄像头统计在主JSON中；同一静态片段的61帧不是61次独立实验，不报告伪精确的独立样本置信区间。

有效深度仍采用原采样合同：原生3×3深度变化≤0.5m、有效1.5–80m、排除边界/无效点。它不是完整图像或所有障碍边缘的质量；GT深度是optical Z，BEV包含XY及高度约束。

## 2. 60–80m误差：偏近主导，分散与多峰并存

| 模型 | Mean MAE | Mode MAE | Median MAE | Mean Bias | 熵 | 分布期望绝对误差 |
|---|---|---|---|---|---|---|
| Joint | 43.821 | 45.502 | 45.676 | -43.821 | 0.821 | 44.309 |
| Far balanced | 35.735 | 35.789 | 36.969 | -35.731 | 0.808 | 36.364 |
| Road point | 31.465 | 31.800 | 32.562 | -31.453 | 0.803 | 32.353 |

| 模型 | GT Bin概率 | ±1m质量 | ±3m质量 | ±5m质量 | 较GT近>5m质量 | 启发式双峰比例 |
|---|---|---|---|---|---|---|
| Joint | 0.83% | 0.62% | 1.65% | 2.68% | 95.23% | 15.82% |
| Far balanced | 1.48% | 1.04% | 2.94% | 4.73% | 92.53% | 27.05% |
| Road point | 2.56% | 1.88% | 5.06% | 7.81% | 88.34% | 40.30% |

Road Point远场GT平均67.69m，预测均值36.24m，Mode均值36.31m。Bias −31.45m几乎等于MAE 31.47m；分布绝对误差32.35m中，低估贡献31.90m，高估贡献0.45m。它主要把远物预测到中近距离，**不是正确远场尖峰被Softmax期望单独拖坏，也不是换Argmax就能修复**。

同时熵0.803、分布标准差15.14m、最大Bin概率均值0.148，分布显著不集中；约40.3%满足上述双峰启发式，但其余也可能宽而偏近。群体平均分布不能证明单射线多峰，因此另附相同GT选点的逐射线概率图。GT Bin中心量化MAE仅0.700m，无法解释31m；本轮没有修改Bin。

未验证的原因包括远场外观退化、数据覆盖不足、损失权衡及场景先验；无法仅凭这三个checkpoint确定它们各自因果贡献。

## 3. 近场退化：偏远、正确Bin质量降低，Town04右侧视角突出

| 模型 | 1.5–10m MAE | Signed Bias | Mode MAE | Median MAE | GT Bin概率 | 熵 |
|---|---|---|---|---|---|---|
| Joint | 1.473 | 1.342 | 1.356 | 1.346 | 0.720 | 0.129 |
| Far balanced | 2.046 | 1.945 | 1.915 | 1.863 | 0.652 | 0.224 |
| Road point | 1.924 | 1.841 | 1.711 | 1.675 | 0.658 | 0.201 |

近场Road Point Bias为+1.84m，主要是推远；GT Bin概率从0.720降至0.658，较GT远>5m概率质量从8.29%升至11.43%。Mode/Median也比Joint退化，说明不仅是少量远尾拉动期望。

训练299样本统计：原像素平均损失下，近场平均权重71.86%、60–80m仅0.628%；四段等权后变为25.08%和24.75%，远场相对上调39.43倍，近场降为原来的0.349倍。这个改变与观测到的权衡一致，但选步及Road监督同时变化，不能声称已完成损失单变量证明。

Town04近场前右/后右MAE：Joint 3.90/3.87m，Road Point 4.76/4.63m；前左/后左则为0.35/0.29m。保存的首帧图像显示右侧近墙外观，错误集中于这些视角；只能作为定位线索，尚未有墙体语义或材质控制实验，不能断言墙面材质就是原因。

## 4. 是否学到视觉信息：有，但泛化增益很弱

| 模型 | 原RGB MAE | 前摄局部遮蔽 | RGB逐图均值 | 特征空间打乱 | 视觉特征置零 |
|---|---|---|---|---|---|
| Joint | 4.151 | 4.207 | 4.611 | 10.584 | 8.343 |
| Far balanced | 4.409 | 4.467 | 4.600 | 11.702 | 17.633 |
| Road point | 4.028 | 4.069 | 5.503 | 13.232 | 24.664 |

遮蔽仅前摄中心宽/高各50%的区域，占该摄像头25%面积；其余五摄、K、外参和所有GT固定。全局MAE变化小不能解释为不依赖RGB：Road Point被遮蔽射线的逐帧平均MAE增量约0.408m，道路IoU从0.541降至0.415。空间打乱与置零是明显分布外输入，证明函数依赖而不能单独证明信息有用。

| 验证场景 | 像素均值先验 | Joint | Far balanced | Road point |
|---|---|---|---|---|
| m4drive_town04_0_retry | 4.547 | 4.567 | 4.959 | 4.497 |
| town03_spawn45_static | 3.358 | 2.932 | 2.796 | 2.650 |

全验证Camera/Pixel Prior MAE=4.245m，Road Point=4.028m；但Town04仅4.547→4.497m，约1.1%改善。BEV内Prior=1.655m，Joint=1.728m，Road Point=2.134m，三者均未超过先验。结论是**存在部分有用视觉学习，不能声称已稳定学到超越位置先验的空间理解**。

| 输入/特征条件 | Joint道路IoU | Far道路IoU | Road Point道路IoU |
|---|---|---|---|
| original | 0.511 | 0.494 | 0.541 |
| front_local_mask | 0.473 | 0.476 | 0.415 |
| rgb_spatial_mean | 0.437 | 0.582 | 0.271 |
| feature_spatial_shuffle | 0.407 | 0.308 | 0.487 |
| feature_zero | 0.574 | 0.345 | 0.320 |
| position_prior | 0.501 | 0.501 | 0.501 |

共同v1掩码的“全预测道路”先验IoU为0.694，三个原RGB模型均未超过；该正类IoU受类别比例影响，不能单独衡量安全通行能力。Joint特征置零和Far逐图均值输入反而得到较高汇总道路IoU，说明类别/位置偏置和场景失配仍很强，不能把“扰动后有变化”当作可靠道路识别。Road Point在Town04优于v1位置先验（0.506 vs 0.456），在Town03反低（0.754 vs 0.786）；两个场景还不足以宣称跨地图泛化通过。

## 5. BEV空间质量与Depth MAE并不一致

当前BEV x∈[-20,60)、y∈[-30,30)、z∈[-2,4)，四高度层。867,467有效点中730,176点位于此三维范围；其中近场658,447点、10–30m 70,332点、30–60m仅1,397点，60–80m为0点（仅XY判断也是0）。不能宣称远场BEV通过，也不能从零样本外推未来场景。

| 模型 | 全范围MAE | BEV内MAE | BEV内同GT体素概率 | BEV内XY≤1m且同高度层质量 | 匹配射线Mass空间TV↓ |
|---|---|---|---|---|---|
| Joint | 4.151 | 1.728 | 0.473 | 0.779 | 0.404 |
| Far balanced | 4.409 | 2.225 | 0.445 | 0.728 | 0.396 |
| Road point | 4.028 | 2.134 | 0.444 | 0.740 | 0.396 |

Joint→Road：全范围MAE改善、BEV内MAE退化，同GT体素质量0.473→0.444、1m邻域同高度质量0.779→0.740；聚合Mass TV略改善0.404→0.396。它们评价不同：聚合质量允许不同射线相互补偿，逐射线位置质量不会。帧级MAE/TV整体相关虽约0.99，主要含两场景差异；不能当作跨checkpoint或因果一致性。

远场虽不在当前BEV内，其错误概率仍投进BEV：60–80m射线落入BEV的平均概率从Joint 48.15%降至Road 28.01%，这是有价值的减少错误投影；所有范围外GT射线对预测总质量的贡献逐帧均值从6.11%降至4.61%。不能因为GT范围外就忽略这些射线。

| 匹配射线投影 | Joint Mass TV | Far Mass TV | Road Point Mass TV |
|---|---|---|---|
| oracle_continuous | 0.000 | 0.000 | 0.000 |
| oracle_gt_bin | 0.273 | 0.273 | 0.273 |
| learned_distribution | 0.404 | 0.396 | 0.396 |
| expected | 0.409 | 0.407 | 0.427 |
| mode | 0.408 | 0.396 | 0.401 |
| median | 0.407 | 0.397 | 0.402 |
| pixel_prior | 0.328 | 0.328 | 0.328 |

TV=½Σ|归一化投影质量−归一化Oracle计数|；另记录未归一化质量比、L1/Oracle点数和各高度层质量，防止只看形状。连续Oracle与既有`observed_counts`逐格完全相同。GT Bin中心Oracle已有0.273 TV，说明0.5m硬格边界及高度分层对量化敏感；这不是修改Bin的证据，且不应将它称为可达到性能下界。像素均值先验投影TV=0.328优于三个模型。

四层Height BEV以相同RGB context、相同精确UV、相同有效射线做连续Oracle/GT Bin/完整概率/Mean/Mode/Median/Pixel Prior对照；输出特征L1和共同支持余弦。完整概率在很多Oracle无点格仍产生特征，所以union特征L1会很大；共同支持余弦高也不代表位置准确。这些特征度量没有语义真值，不算Occupancy质量。

Surface Head预测的是log(1+可见采样点数)。原始输出全格MAE：Joint 0.100、Far 0.120、Road 0.132；有GT点格为0.292/0.323/0.288。大量零采样格影响全格分数，且约26–32%输出为负log-count，当前仅作无约束辅助回归，不当作物理占用概率。

### 排除有效射线筛选混杂的补充实验

只用有效GT射线会改变上下文覆盖；不能把该Oracle道路输出当成模型上界。补充实验保留原生全部射线、RGB与context，对有GT位置替换深度概率，其他位置保留原分布。原生射线与最近原生GT像素的最大偏移0.3334个缩放后像素；因此这里只称GT Bin替换，不假称连续精确Oracle。

| 原生全射线条件 | Joint道路IoU | Far道路IoU | Road Point道路IoU |
|---|---|---|---|
| original | 0.511 | 0.494 | 0.541 |
| depth_probability_shuffle | 0.510 | 0.492 | 0.541 |
| context_spatial_shuffle | 0.409 | 0.311 | 0.492 |
| oracle_gt_bin_valid_only | 0.574 | 0.469 | 0.576 |
| pixel_distribution_prior_valid_only | 0.503 | 0.483 | 0.540 |

深度概率沿像素位置打乱后，道路IoU几乎不变；context打乱影响大得多。Road Point深度打乱后BEV相对L1变化约0.094、道路概率变化0.0166，context打乱对应1.035和0.381。**道路头对当前准确深度投影的依赖不足**，但这是有限扰动集合的证据，不是证明深度永远不重要。

Road Point保留全部射线的GT Bin替换使Town04 IoU从0.506升至0.555，Town03从0.754降至0.701；Far总体也下降。冻结头在分布外几何输入上未必受益，不能用这个Oracle直接预测重新训练后的上限。有效射线版本Road IoU约0.172，与全射线替换0.576差异巨大，必须保留这项覆盖控制。

代码层面：HeightSplat输出Σ(p·f)/max(Σp,1e−6)，mass另作为4通道输入。若一个格只有单射线贡献且mass>1e−6，其特征就是f，概率幅值会抵消；因此微小概率也能填入视觉特征。合成测试确认这一性质。原生Road中mass在(1e−6,1e−3)的格仅含总质量0.015%，却占约7.15%的Height特征绝对值；Joint约0.028%质量对应21.97%特征绝对值。该统计受阈值和特征尺度影响，只提示应审查置信度传递，尚未证明某种门控修复一定有效。

## 6. 静态/驾驶差异：控制到哪里，不能归因什么

| 验证组 | 帧数 | 运动 | 地图是否进入训练 | 天气 |
|---|---|---|---|---|
| Town04驾驶 | 161 | 155帧>1m/s；6帧≤1m/s | 未进入训练 | fog_density40，sun50，wetness30 |
| Town03静态 | 61 | 全部≤1m/s | 同地图其他路线进入训练 | fog_density2，sun30，wetness0 |

每个扰动实验均在同一帧内对照，因此地图、天气、姿态与GT固定。跨模型也固定同一验证样本，但训练因素如上表不同。跨场景则无法同时固定地图、天气、道路宽度/墙体布局和运动。Town04最初6个低速帧仍难（Road MAE4.545m），与后155个运动帧4.496m接近，不能把差异简单归为运动模糊；两组位置/时间仍不同，且没有速度干预。

另按共同摄像头×GT距离段重新加权：仅保留两场景均≥100点的23个分层，权重取两场景较小计数，消除一部分相机/距离构成差异。

| 重加权模型 | Town04 MAE | Town03 MAE | Town04先验 | Town03先验 |
|---|---|---|---|---|
| Joint | 3.407 | 1.761 | 3.291 | 2.038 |
| Far balanced | 3.698 | 1.916 | 3.291 | 2.038 |
| Road point | 3.611 | 1.931 | 3.291 | 2.038 |

控制构成后差异仍存在，且Town04模型弱于先验。这是描述性标准化，不是匹配地图/天气的因果实验，也不是新的正式验证分数。缺少同地图同路线同天气的静止/运动对照和同路线换天气序列；应优先采这些配对数据，而非随机拆分相邻帧。未见地图仅一个驾驶片段，没有多seed或独立路线重复验证。

## 7. 回答项目决策问题

1. **是否真正学到超越先验的视觉信息？** 部分学到：遮挡、均值图和特征扰动造成明确变化，Town03及全验证深度有收益；Town04收益弱，BEV内几何和聚合Mass仍输给像素先验，不能称可靠泛化。

2. **31m远场误差来源？** 当前直接证据是系统性低估为主，并伴随高熵和部分多峰。Mode/Median并不能消除误差，Bin量化只约0.7m。导致这些现象的训练/数据原因尚不能单变量归因。

3. **近场为何退化？** 近场预测被推远，GT Bin质量下降；距离损失相对权重大幅转向远场与此一致。右侧近墙视角最突出；选步、道路标签及权重混杂仍在，需受控短实验而不是继续加步数。

4. **Depth MAE与BEV质量一致吗？** 不一致。全MAE、BEV内MAE、逐射线概率落点、聚合Mass及冻结道路头分别出现不同排序；后续门禁必须同时看这些指标及先验，不能靠全局深度MAE选择系统方案。

5. **下一步优化Depth还是推进Occupancy/Temporal？** 不把工作收缩成单独深度估计竞赛。保留Depth链路，先验证概率到BEV的置信度传递及范围内几何；并行推进Occupancy监督合同/Oracle标签与Temporal几何接口验证。当前不适合直接上大型Occupancy/Temporal训练，更不能把时间融合视为能修复偏差的保证。

推荐顺序（尚未执行）：①在固定checkpoint上对mass门控/归一化做小范围推理诊断，同时检查误投影质量与道路输出；②补同路线天气/运动配对数据、近墙/坡道/交叉口及独立地图，保留盲测；③建立Occupancy Oracle与可见性合同；④验证ego-motion warp、时间一致性、动态物体/遮挡处理，再开展小型Temporal基线。若随后需要训练，应先冻结BEV相关门禁和近场不退化约束，用有限单变量实验决定，而不是本轮自动开新训练。

### Occupancy监督边界（拟定合同，未实现/验收）

- **Occupied**：传感器确实观测到的物理表面落入体素，附语义/高度；体素表面定义下地面也是表面，后续“道路上的障碍占用”需显式区分ground、Drivable与障碍，不能混为一个二值标签。表面厚度/定位容差需预先固定。
- **Free**：通过可用深度射线的显式ray traversal，在首个可靠表面前、扣除表面容差区后确认穿过的体素；自车、传感器盲区、无效/截断/无回波射线不能自动标Free。
- **Unknown**：未观测、遮挡后方、范围外、深度无效以及冲突未解决区域；默认忽略监督。合并多摄时表面占用优先，冲突保留质量标识；不沿用“无点=Free”。
- 本轮只有稀疏可见Surface Count与地图静态道路监督，未生成上述Free/Occupied/Unknown GT，未做语义Occupancy或闭环成功声明。

## 8. 产物、测试与复现

- [完整分布/分摄/距离/场景/BEV内诊断JSON](m4_depth_distribution_diagnostics.json)
- [BEV逐帧汇总与图像哈希](m4_depth_distribution_bev_summary.json) · [原生全射线干预](m4_bev_full_ray_interventions.json)
- [场景构成标准化](m4_diagnostic_scene_controls.json) · [最终数据/模型完整性审计](m4_depth_distribution_audit_v2.json)
- [解码、熵及偏差](m4_depth_distribution_figures/depth_decoders_and_uncertainty.png) · [概率窗口质量](m4_depth_distribution_figures/gt_neighborhood_mass.png)
- [逐射线远场概率图](m4_depth_distribution_figures/far_probability_examples.png) · [Depth与BEV空间质量](m4_depth_distribution_figures/depth_vs_bev_spatial_quality.png)
- [四层BEV对照](m4_depth_distribution_figures/m4_road_point_m4drive_town04_0_retry_bev.png) · [RGB与特征扰动](m4_depth_distribution_figures/rgb_feature_interventions.png)
- [Town04六摄GT/预测/熵](m4_bev_full_ray_figures/m4_road_point_m4drive_town04_0_retry_cameras.png) · [全射线道路对照](m4_bev_full_ray_figures/m4_road_point_m4drive_town04_0_retry_road.png)

测试：`tests/test_perception_diagnostics.py`覆盖单峰/均匀/双峰、Median/窗口积分、偏差符号、分组掩码、BEV半开边界、范围外rank别名、概率质量和特征归一化、扰动与hook清理。完整GPU环境测试112 passed, 5 skipped；结果见`logs/m4_diagnostics_final_tests.log`。

审计保留一次失败：首次要求原生复算道路混淆矩阵完全一致，Joint/Far各有至多2个阈值格差异（约163万有效格中），IoU最大差7.55×10⁻⁷，与CUDA scatter/BF16阈值数值扰动相符，未逐格证明根因。保留`m4_depth_distribution_audit.json`失败记录，v2明确采用≤2×10⁻⁶ IoU且混淆矩阵L1≤4的容差，不声称位级确定性。另加范围外GT rank防别名硬化；原结果非负质量汇总证明无受影响样本，不覆盖旧报告。

下面命令只做读取/诊断，输出均指定新路径；每个工具拒绝覆盖自己的输出。先前报告、checkpoint和索引保持原样。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
.venv-gpu/bin/python -m pip install -r requirements-diagnostics.txt
.venv-gpu/bin/python -m pytest -q tests/test_perception_diagnostics.py
.venv-gpu/bin/python -m tools.diagnose_m4_perception \
  --output reports/m4_diag_repeat.json --artifacts reports/m4_diag_repeat_artifacts
.venv-gpu/bin/python -m tools.report_m4_perception \
  --input reports/m4_diag_repeat.json --figures reports/m4_diag_repeat_figures \
  --summary reports/m4_diag_repeat_bev.json
.venv-gpu/bin/python -m tools.diagnose_m4_bev_interventions \
  --diagnostics reports/m4_diag_repeat.json --output reports/m4_fullray_repeat.json \
  --artifacts reports/m4_fullray_repeat_artifacts
.venv-gpu/bin/python -m tools.plot_m4_full_ray \
  --input reports/m4_fullray_repeat.json --output reports/m4_fullray_repeat_figures
.venv-gpu/bin/python -m tools.audit_m4_diagnostics \
  --main reports/m4_diag_repeat.json --supplement reports/m4_fullray_repeat.json \
  --output reports/m4_diag_repeat_audit.json
.venv-gpu/bin/python -m tools.write_m4_diagnostic_analysis \
  --input reports/m4_diag_repeat.json --bev reports/m4_diag_repeat_bev.json \
  --supplement reports/m4_fullray_repeat.json --audit reports/m4_diag_repeat_audit.json \
  --output reports/m4_diag_repeat_analysis.md --controls reports/m4_diag_repeat_controls.json
```

精确依赖锁位于`m4_depth_distribution_artifacts/environment.lock`；新增绘图依赖matplotlib3.10.7。主实验协议包含原有报告/模型哈希、配置、输入哈希及代码指纹。模型代码、训练配置和数据接口没有改动；诊断模块独立于推理/训练路径。

方法背景：[Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711)以深度分布提升视觉特征并聚合到BEV；[BEVDepth](https://arxiv.org/abs/2206.10092)强调显式深度监督与投影质量。本文只借用其问题框架，所有数值和判断来自本项目实测，不将论文结果当作本系统验收证据。
