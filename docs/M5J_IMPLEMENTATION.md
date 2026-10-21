# M5-J — Mobile Visual Migration & BEV Evidence Calibration

基线 `3c9c8ad`，分支 `feat/m5j-mobile-bev-evidence`。历史模型、数据、报告和默认驾驶代码只读。所有新增源文件和实验使用 m5j 前缀。

## 推理与证据合同

显式 `MobileBEVCandidate(enable=True, visual='mobile_gt'|'legacy', seed=101|102)`；未enable直接拒绝。RGB分别进入固定Mobile/M5G Pixel分支及原ResNet Depth分支，不交叉使用Mobile特征。所有权重版本和完整SHA由`configs/m5j_models.json`验证。统一输入白名单沿用项目合同，限定B1六摄384×640、合法K/T、固定相机顺序、有效相机和正时间戳；调用方可提供now_s执行未来/超过250ms陈旧拒绝。GTDepth、Actor、Instance、PairWitness、ValidMask不能进入Candidate forward。没有车辆控制输出。

输出：原图Pixel Foreground、原48LID DepthProbability、四层160×120证据与支持字段、原始时间戳。Pixel采用历史floor256量化用于严格对照；BF16视觉与FP32几何分离。raw与旧m5f_lift数值误差在1e-6以内，旧实现未修改。返回多个研究分数，不默认挑选某个候选，更不将其当校准Occupancy。

独立射线按(camera,feature_pixel)去重；同一射线多个bin进入同一个凸体素只计一次。单条直线按深度单调进入凸体素，连续同rank去重与显式unique合成对照。ray_bin_count、ray_count、camera_count分列。统计独立性并不成立，不能把相邻射线当独立场景。几何支持包括微小概率，质量另外报告；最大单射线质量按(ray,cell)真实聚合。相机质量、总质量、质量加权归一熵也保存。

## 预声明对照

A raw=`1-exp(-Σqp)`；B gate固定二维q≥.9再raw；C regularized=`raw*mass/(total+1)`，保证不放大小质量、score≤raw。完全固定Pixel/Depth/P8中心4::8/K/T/网格。阈值集合见`configs/m5j_protocol.json`，先在Val筛选相对raw@.2的+5pp精确表面Precision、同层1m Recall最多降2pp、Actor检出最多降2pp，两个seed同时通过。Val选择工作点、Recall匹配和Precision匹配，Test只执行，不再选择。PR曲线是描述性结果。

D仅当C在两个seed场景平均AP均超过raw时启用；实际不满足，因此未训练D。没有视觉/Depth更新，没有额外Loss，也未用历史Test选择新参数。没有把降低分数获得的背景FPR改善当成功。

Reference复用可靠全分辨率首表面：target/confirmed-background/unknown分离，混合粗格排除；同层空间匹配为多对一表面覆盖，不是物体一对一检测。Actor参考由renderer instance与近似AABB关联，不是假定碰撞盒=mesh。四种Oracle干预统一P8有效Mask，正式双分支无GTMask，不能把Oracle分支当正式推理。

来源审计A/B/E按贡献射线源GT拆分；C沿深度扩散、D多射线累积可重叠。Unknown源不能裁决二维正确或错误；Unknown目标格不计物理FP。trace保存来源camera/UV、完整48bin分布、目标深度、体素三维范围、质量、相机分项及参考。总体归因使用全部高分格，代表图使用固定最高分规则，不能拿代表图计发生率。

## 数据与采样

共享M5I Train/Val只读；本轮无训练，无新增Train/Val。已打开M5ITest仅用于相机/小目标历史回归。新Test四个独立位置，各20配置；地图几何优先近路口，距全部历史位置和M4路线至少45m。模型/阈值/源码冻结后才采集。每个配置包含六摄RGB/Depth/Instance同步及自车移除参考，Depth/Instance仅监督。位姿与时间戳、校验和、RGB跨集合重复另检。

S4/P8/S16及phase0/center是Oracle采样覆盖诊断，没有新的S4Depth网络。Mobile RGB高置信ROI门控后仍以全目标参考计算Recall，保留被ROI遗漏部分。Replay只读评估，不升级默认；严格3×3深度连续Mask的20–35m关联Actor射线独立列出。

## 实际复现顺序

所有目标默认拒绝覆盖。重跑使用独立工作树和新的输出命名；原始缓存、失败日志及源码版本保留。

```bash
.venv-gpu/bin/python -m tools.m5j_prepare
runuser -u ubuntu -- bash scripts/start_carla.sh /root/gpufree-data/simulators/CARLA_0.9.15 2100 Epic
.venv/bin/python -m tools.m5j_capture_protocol
.venv-gpu/bin/python -m tools.m5j_cache --split val
.venv-gpu/bin/python -m tools.m5j_gpu_contract
.venv-gpu/bin/python -m tools.m5j_evaluate --split val
.venv-gpu/bin/python -m tools.m5j_population --split val
.venv-gpu/bin/python -m tools.m5j_spatial_details --split val
.venv-gpu/bin/python -m tools.m5j_depth_sampling --split val
.venv-gpu/bin/python -m tools.m5j_camera_quality --split historical
.venv-gpu/bin/python -m pytest -q -rs
.venv-gpu/bin/python -m tools.m5j_freeze
.venv/bin/python -m tools.m5f_capture --protocol configs/m5j_capture_test.json --output /root/gpufree-data/hs-drive-data/raw/m5j_sealed
.venv/bin/python -m tools.m5a_dataset --raw /root/gpufree-data/hs-drive-data/raw/m5j_sealed --output /root/gpufree-data/hs-drive-data/processed/m5j_test_index --report reports/m5j_evidence_dataset.json
.venv/bin/python -m tools.m5g_dataset --indices /root/gpufree-data/hs-drive-data/processed/m5j_test_index --output /root/gpufree-data/hs-drive-data/processed/m5j_test_pixels --splits test
.venv-gpu/bin/python -m tools.m5j_dataset_audit
.venv-gpu/bin/python -m tools.m5j_cache --split test
.venv-gpu/bin/python -m tools.m5j_evaluate --split test
.venv-gpu/bin/python -m tools.m5j_spatial_details --split test
.venv-gpu/bin/python -m tools.m5j_population --split test
.venv-gpu/bin/python -m tools.m5j_depth_sampling --split test
.venv-gpu/bin/python -m tools.m5j_camera_quality --split test
# Stop this task's CARLA and all other GPU jobs before isolated timing.
.venv-gpu/bin/python -m tools.m5j_latency
.venv-gpu/bin/python -m tools.m5j_figures
.venv-gpu/bin/python -m tools.m5j_finalize
.venv-gpu/bin/python -m tools.m5j_artifacts
```

实际初次Val缓存完成后写审计JSON遇到NumPy bool序列化，修复仅写出转换并使用`--audit-only`重新生成诊断，原缓存不覆盖。初次评测完成主指标后，零预测工作点Precision=None造成选择器失败；保留主指标JSON，从同一aggregate用修复后的`select()`恢复，未改变阈值集合和选择标准。初始源版本见`reports/m5j_logs/source_versions`。

独立GPU全Candidate计时含输入验证、Pixel、Depth、Lift、支持计数，单独报告CPU读图/校验/解码、序列化、反序列化、H2D。未执行在线Shadow，无Tick Owner改动；不将离线GPU延迟称为20Hz驾驶。Runtime继承Partial，Oracle驾驶未重验，纯视觉安全未验证。
