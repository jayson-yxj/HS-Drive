# M8-J3 单片段下载与真实数据验证

基线：c4679352669e3a2b3add0ccb0fcc05282e9285d6。最终范围依据本轮后续用户明确授权：个人非商业研究，仅下载指定旧 Mini 片段、本地解压、读取与可视化。不公开数据或衍生图像，不做模型前向/训练/CARLA。未修改默认模型、标定、驾驶或历史报告。源码与结果由 m8j3_release_manifest.json 和提交绑定。

## 执行与本地存储

官方 HF 仓库 rethinklab/Bench2Drive，revision ce56cfbc26edf7796586e26fd44644f0ac2f6bcc，文件 DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz。使用独立环境 huggingface_hub 0.26.5 的 hf_hub_download，本机 HTTP 代理127.0.0.1:7897、匿名 token=False、TLS校验、15s元数据超时、30s读取超时、最多2次外层尝试及每次最多5次官方流式恢复，外部1200s总时限。官方客户端自动保留.incomplete支持续传；本次首轮完整成功，没有实际触发恢复。启动账本防止误重复执行，签名URL查询参数不记日志。

下载前可用177068138496 bytes，门槛12GiB。归档预算：8GiB展开、100000条目、128MiB单文件。实际gzip CRC及完整流通过，6237条目（31目录、6206文件），普通文件合计329374497 bytes。禁止绝对路径、父路径、反斜线、控制字符、重复规范路径、符号/硬链接、设备及sparse条目；全部预检后才手动复制。未调用extractall，未执行包内代码，未恢复属主/权限。

私有根目录为 /root/gpufree-data/hs-drive-private/m8j3-bench2drive-single-clip，umask077。download保存归档及官方缓存；extracted保存原始数据；validation保存局部诊断、提取文件SHA和可视化；rgb_only保存三个导出样本。均在Git工作区之外。只提交统计、字段结构、SHA及源码，不提交逐对象标注、样本、图片或checkpoint。

## 最小加载合同

新增 datasets/m8j3_rgb_clip.py，独立于M8-B和默认数据路径。私有预处理仅从标注导出静态K、cam2ego和RGB引用；读取器只访问RGB根目录，不读anno、Depth、Instance、Semantic、World Transform、Actor状态或GT mask。代码/路径隔离不等于同一Unix用户下的OS沙箱；未来训练进程应只挂载RGB目录和单独监督提供器。

HS固定顺序 front/front_left/front_right/rear_left/rear_right/rear，对应 CAM_FRONT/CAM_FRONT_LEFT/CAM_FRONT_RIGHT/CAM_BACK_LEFT/CAM_BACK_RIGHT/CAM_BACK。不得用字典偶然顺序或另一模型顺序。输出RGB uint8 [6,900,1600,3]、K FP64 [6,3,3]、T FP64 [6,4,4]；本轮没有resize、归一化或模型执行。

光学基为右/下/前，CARLA为前/右/上，HS为前/左/上。T_hs_from_optical = diag(1,-1,1,1) × cam2ego × optical_to_carla_camera。world2cam为CARLA相机基，不可直接光学投影。使用包内K：主点(800,450)，不可用HS的(W-1)/2重建；后摄FOV110，其余70，安装参数与HS不同。未来resize须定义像素中心约定，K'=A K并保存A。

source_frame与timestamp_s为null，temporal_valid=false；ordinal仅文件序号。不得伪造World Frame或elapsed_time=ordinal/10。投影叠图是几何诊断，不证明实例关联、真实同步或全部标注可见。

## 复现

以下命令只读复核或合成测试，不再下载：

    cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
    .venv/bin/python -m pytest tests/test_m8j3_archive.py tests/test_m8j3_rgb_clip.py -q
    .venv/bin/python -m tools.m8j3_verify_payload
    python3 -m tools.m8j3_review verify

已经执行的写入命令（账本和独立目录禁止覆盖；不要删账本触发重复下载）：

    timeout --signal=TERM 1200s /root/gpufree-data/hs-drive-experiments/m8j2-metadata-env/bin/python -m tools.m8j3_download_clip
    python3 -m tools.m8j3_archive_audit
    .venv/bin/python -m tools.m8j3_validate_clip
    .venv/bin/python -m tools.m8j3_supplement

需迁移数据路径时保留相同SHA并显式调整路径常量。后续可选GPU研究不在本轮执行命令内。

## 结果与未通过项

79项既有离线合同测试（排除梯度测试）与20项新增归档/加载测试通过；归档与图像可读、静态几何、三帧RGB隔离通过。真实CARLA、GPU前向、训练、时序同步未执行。

初次全组文件对齐复核失败：expert_assessment为-0001…00212。原失败日志保留；固定官方采集源码save处用frame-1与现象一致，但未验证专家内容的真实语义对应。复核器显式确认这一失败并隔离，不自动shift。主图像与anno为00000…00213，不能由此认证时间同步。

Depth uint8不可恢复原24bit浮点；Instance尚未可靠关联Actor；隐藏目标、静态背景、有效可见范围及缺失标签不能自动提供全负样本。只有一个相关路线组，禁止按帧随机划分后称独立验证。正式训练、Occupancy、Temporal、安全门禁保留。
