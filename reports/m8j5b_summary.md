# M8-J5B — Coverage-Aware Supervision

日期：2026-10-26 UTC。基于 `6ee9bf7af7b28910def0efb9693c75790a357356`，独立分支 `feat/m8j5b-coverage-aware-supervision`。实际完成 CPU 214帧覆盖审计、合成CUDA梯度测试、四帧冻结RGB前向后的Loss诊断与私有可视化。没有模型反向、优化器更新、CARLA启动或新数据下载。人工签核仍 Pending / Suspended。

**结论：工程实现可以阻断未知目标的错误背景梯度，且合成安全负样本仍能正常学习；真实片段尚无安全负监督证书，当前不具备启动真实梯度Smoke的完整条件。**

## 缺标区域依据与覆盖

固定CARLA官方语义源码核实 pedestrian12、car14、truck15、bus16、motorcycle18、bicycle19；也对rider13、train17保守保留。Bench2Drive采集器将Semantic R通道保存为单通道，将Instance原始BGRA保存。Instance G/B组合只是截断渲染token，不能关联为Actor ID。来源与证据等级见 [实现说明](../docs/M8J5B_IMPLEMENTATION.md)。

全片段六摄1284个视图的Semantic/Instance全像素一致率87.38%，相关前景并集内95.82%。这只能支持分割区域诊断，不是独立物理真值；当前实际出现的目标语义只有car/person，其余类别路径由合成测试覆盖。

每个同语义render区域与对应类原始Box投影比较；少于50%空间覆盖作为缺标候选。另保留所有未覆盖前景像素、跨分割不一致像素和不支持类，并在原始图上外扩4px。没有最小面积丢弃规则。用area resize保留任意像素支持，再应用既有PETR固定crop。

- 缺标/不确定候选635个区域视图观测，其中≥100px有321个、分布164帧；其余314个小区域同样进入Ignore。321比J5A的320多1，是未获排除授权的ego可能区域；不把它认证为新物理漏标。635不是635个独立目标，包含连续帧、视角及微小不确定性。
- **局部Ignore占原始图像0.1433%，占PETR实际crop 0.2386%。** 这是像素占比，不等于Query忽略率，也不证明其余像素是安全背景。
- 四个指定缺标案例全部复现并覆盖，包括00124可见行人。四例语义一致率分别为99.966%、92.199%、88.197%、88.444%；后两例不足以可信正标，但必须保持Ignore。
- 没有产生任何缺标对象的新3D Box，没有使用旧Depth uint8恢复深度或推断安全Free。

## 正监督资格与真实Query

现存源Box中自动满足几何、类别、模型域和可见语义支持的诊断候选为 **570条：car499、pedestrian71**。相比J5A旧技术候选545，新增25条全部来自旧render身份冲突隔离；它们仍保持身份不确定，不改变源Box，也不声称人工通过。25条逐条原因保存在私有目录。这是Positive资格维度拆分，尚未批准训练。

冻结模型实际四帧，batch1、900 queries、六层各自匹配：

| 文件序号 | 每层 Positive | 每层 Ignore | 每层 Negative | 最终层局部图像Ignore触及Query |
|---|---:|---:|---:|---:|
| 00107 | 4 | 896 | 0 | 164 |
| 00124 | 3 | 897 | 0 | 129 |
| 00196 | 2 | 898 | 0 | 104 |
| 00213 | 2 | 898 | 0 | 131 |

四帧六层共21600个query-layer观测：P=66、I=21534、N=0；Ignore为99.694%。这不是21600个独立预测对象。所有层都报告HighIgnoreFraction和NoNegativeSupervision。局部Ignore与最终帧级Ignore分开报告；Positive即使投影触及Ignore仍受保护。

本片段缺少全十类、遮挡空间及时间一致性的安全负证据，214帧全部保守禁用未匹配背景监督。**不允许以0.2386%的局部mask声称剩余99.7614%的输入都是可靠负样本。** 若停留在这个状态进行训练，会失去正常背景学习，所以真实更新Smoke和正式训练均保持No-Go。可以继续做不更新权重的损失诊断。

## 实现与验证

新Loss显式调用，不修改官方PETR或默认训练路径。所有Decoder层一致应用Positive优先、图像Ignore否决、3D安全证书和帧级退化。完整模式与官方数值/梯度一致，Unknown零梯度，安全Negative有正确梯度，坏GT不进回归，全Ignore/零Positive有限。

- 32项CPU测试通过（含历史回归和真实缺标案例，0跳过）。
- 31项实际RTX4090/PETR环境测试通过（15项新测试、16项历史回归）。仅合成预测张量反向，不更新模型。
- 真实私有页面9项浏览器断言通过，浏览器禁网；四例对照已实际查看，没有人工签核。
- 四帧冻结前向总调用4次，模型参数逐张量校验未变。后三帧前向分别696.02/518.34/520.37ms，峰值allocated约1.030GiB；首帧已完成前向后因投影校验失败，复用其缓存，不补造缺失耗时。Loss审计约50.68–214.42ms，含首次开销；这是离线研究测量，不是实时预算通过。

失败记录全部保留：旧Torch不支持tuple-dim `all()`，改为顺序归约；两个真实远车分割一致率未达0.9，撤掉不恰当的“所有缺标都应可靠”断言，保留实际不确定性；监督使用world矩阵消除Ego时存储舍入误差被K放大，改为与RGB输入完全一致的cam2ego路径，未改变旧数据/像素mask或放宽阈值。最终四帧矩阵差≤0.00005922。

## 下一步门禁

**CoverageAwareLossEngineeringGo / SyntheticGradientGo / RealSupervisionPartial / RealGradientSmokeNoGo / TrainingReadinessNoGo。**

需要先建立一个小范围、可核验、与十类PETR损失一致的真实负监督来源或明确的类别级监督合同，并对真实混合P/I/N执行无更新检查；不以分割“没有前景”或低预测置信度认证空闲。新增25条Positive仅是诊断候选，其源几何/语义支持策略应在下一实验协议中明确冻结。随后才可单独批准1–2帧、≤3步的梯度Smoke，检查负学习仍然存在，不把人工签核重新当成修补漏标的替代方案。

产物：[覆盖审计](m8j5b_coverage_audit.json)、[Loss审计](m8j5b_loss_audit.md)、[真实Query统计](m8j5b_real_query_diagnostic.json)、[验收](m8j5b_acceptance.json)。私有可视化：`/root/gpufree-data/hs-drive-private/m8j5b-coverage-supervision/web/index.html`，包含原始GT、转换GT、正诊断资格、缺标候选、局部Ignore与最终帧级退化的独立开关。
