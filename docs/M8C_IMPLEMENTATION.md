# M8-C 实施与复现

基于 `af4266be5859e55db51e77e4b1b15ef122fce61f`，分支 `feat/m8c-petr-reference-distribution-audit`。任务限定诊断，不训练、不更改默认推理、感知或控制代码。已有M8-B 192帧/38位置/57600候选只读。冻结历史manifest1620项和Evaluator参考1735项SHA全部核验。

## 文件及执行隔离

- `configs/m8c/protocol.json`：预声明12个Hook输入、26个背景审计帧和身份支持规则；补充样本固定于检查前。Train/Val用于开发审计；Calibration和旧Sealed只执行冻结规则的历史描述统计，无参数/阈值选择。
- `tools/m8c_gpu_hook.py`：复用M8-A隔离Torch2/cu118环境，numeric NPZ allow_pickle=False严格加载558张量；不使用Actor/Depth/Instance。网络namespace关闭、Python open审计拒绝GT目录，环境清理。可信源码观测隔离，不称防恶意native代码文件系统沙箱。
- Hook仅在官方position_embeding指定行读实际normalized XYZ、depth、img2lidar，返回时读geometry mask，并用position_encoder prehook保存实际inverse_sigmoid输入；保存全部六层900query回归/分类张量。无替换网络函数或参数。首帧3组Hook开/关比较及12帧对M8-B输出逐位一致，结束权重逐位不变。返回张量和原frame/timestamp保留。
- `diagnostics/m8c_geometry.py`：合成参考变换数学接口。相机 `T_R_C=T_E_R^-1 T_E_C`，中心/角点T_E_R变回，速度只旋转。yaw仅适用保持竖直的参考旋转；倾斜参考显式拒绝，不压成虚假yaw。没有物理B标定，所以这个接口未用于GPU对照。
- `tools/m8c_association.py`：独立Evaluator，实例像素、语义、Depth只走评价；旧92个score0.1投影配对保留。多视角身份分级绝不按GT位置误差挑候选。新严格误差只存在于6条IdentitySupported记录，其余metric_error=null。该名称表示自动证据支持，不是认证人工身份。
- `tools/m8c_reference.py`：固定源码/真实相机审查、实际张量分析、B门禁。不安装数据、不猜参考高度。官方页面资料只作参考；schema/tutorial中的真实传感器token可追溯，未取得完整LiDAR标定。
- `tools/m8c_figures.py`：全部154个固定背景候选生成13页，另含全部6个身份支持、8个固定远场代理及统计图。PNG是诊断可视化，不用生成式图片。`m8c_visual_review.json`说明AI助手实际看过图页；不是人类双人标注，不升级规则。

## 监督和评价定义

历史可靠可见性：唯一render ID几何关联、Depth[1.5,80)且>=3 native像素；原缓存像素数逐项对齐。裁剪保留native行v>=151（对应名义332/2.2），边界插值存在不足1px不确定。GT是记录Actor OBB中心，非mesh接触真值。

IdentitySupported：已有score0.1全局一对一投影配对，至少两个可见相机同时满足GT关联purity>=0.9、裁后>=8像素、预测投影矩形对tight实例bbox IoU>=0.5、实例mask召回>=0.8；已记录其他Actor的bbox竞争IoU<0.1。竞争达到门限→AmbiguousAssociation；不足→ProjectionProxyOnly；原规则无配对→NoSupportedAssociation。该保守规则会排除真实单视角目标，不代表它们不是物体或RGB不可辨；不能用6个样本概括所有模型性能。未记录背景身份不能被竞争检查覆盖。

指标只对IdentitySupported计算XY/XYZ、光学深度/横向有符号与绝对误差、径向/Z、尺寸、yaw/轴向yaw、upright IoU；Cone的yaw无明确方向意义置空。未改变M8-B自然检测45/155分母。92代理关联的旧误差仅历史引用，不能作为本轮可信身份定位结果。IoU关联分层另见association_strata；仅1个多Actor帧，0个检测到的竞争歧义不能当歧义不存在。

背景固定Train/Val分层26帧，每帧每粗类取最高3个未匹配score>=0.1，最多12，共154。属于有目的诊断样本，不能外推11045未匹配总体。重复是几何box完全相同（可能同query多类别），不表示已确认同物理Actor。UnsupportedClass专指barrier缺少对应记录标注合同，PETR本身支持barrier。ConfirmedUnmatchedBackground保持0，因为仅投影到路面/墙体不足以证明该3D空间不存在物体。

实际SDK CARLA0.9.15 enum与版本网站展示的legacy语义表不同；以本地SDK、记录像素和历史项目合同为依据。初稿barrier语义重叠字段误用[2,3]，已在代码修为Walls4/Fences5/GuardRail28，**旧计算记录保留且字段更正写在 `m8c_annotation_semantic_correction.json`**。不影响原始histogram、身份分级和154候选状态。报告使用该更正作为权威值。初次合成速度旋转测试抓到转置错误，修复后通过；该代码从未用于物理B模型。初次reference报告重复training_steps关键字失败，保留日志后按相同数据恢复，不覆盖已写内容。

## 实验路径、时间及复现

数据输出：`/root/gpufree-data/hs-drive-experiments/m8c-reference-audit`。所有.npy/.npz/.json/官方下载都外置，仓库报告提供SHA。GPU沿用 `/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/env/bin/python`，不动主环境。命令日志每条含源码SHA、返回码和运行秒数。

以下命令在新checkout及新版本root首次执行；现有已封存报告采用open('x')，不会覆盖。不要删除历史输出来重跑。重现实验复制protocol改root为新目录，并保存原/新protocol SHA；运行新结果后比较冻结旧资产。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
python3 - <<'PY'
from tools.m8c_common import *
run('gpu_hook', ['unshare','--net',str(RUNTIME/'env/bin/python'),'-m','tools.m8c_gpu_hook'])
run('association', [str(RUNTIME/'env/bin/python'),'-m','tools.m8c_association'],timeout=1200)
run('geometry_tests_v2', [str(RUNTIME/'env/bin/python'),'-m','unittest','discover','-s','tests','-p','test_m8c_geometry.py','-v'])
run('reference_analysis', [str(RUNTIME/'env/bin/python'),'-m','tools.m8c_reference'])
run('figures', [str(RUNTIME/'env/bin/python'),'-m','tools.m8c_figures'])
run('supplement', [str(RUNTIME/'env/bin/python'),'-m','tools.m8c_supplement'])
PY
```

实际官方资料获取：urllib经历史proxy拒绝后，使用直接HTTPS（证书验证开启），固定URL/结果在official_access_direct/final.json。下载超时原日志保留；没有无限重试、账户绕过或第三方标定。复现时可直接核验本轮保留的官方页面快照与SHA；不要求联网动态网页保持相同字节。`m8c_semantic_schema.json`由`.venv/bin/python`读取installed carla.CityObjectLabel生成，无连接CARLA server。

验证当前交付可直接运行：

```bash
python3 - <<'PY'
from tools.m8c_common import *
run('verify_new_tests', [str(RUNTIME/'env/bin/python'),'-m','unittest','discover','-s','tests','-p','test_m8c*.py','-v'])
run('verify_m8b_tests', [str(RUNTIME/'env/bin/python'),'-m','unittest','discover','-s','tests','-p','test_m8b*.py','-v'])
PY
.venv-gpu/bin/python -m pytest tests/test_m8a_p1.py tests/test_m8a_assets.py tests/test_m8a_proxy.py tests/test_m7a_shadow.py tests/test_m5o_benchmark.py -q
```

本次13个M8-C测试、18个M8-B合同测试及42个历史接口测试通过。没有新增依赖、训练、CARLA采集或闭环。Hook墙钟~506–512ms，当前同进程无Hook~503ms，差4.6–10.1ms；比M8-B历史53ms慢的跨会话原因未单独归因。该审计含路径监控、同步和张量复制，**不替换M8-B时延或认证实时性**。

## 封存

`reports/m8c_release_manifest.json`包含代码/配置/报告/图/日志/原始Hook和官方证据SHA；引用旧M8-B冻结manifest及Evaluator参考manifest。最终Git提交封存manifest，manifest记录父提交及实际源码SHA避免自引用。所有默认代码不变，原报告和失败记录保留。

## 报告阅读勘误

`m8c_depth_lateral_error.json` 的 association_scope 早期描述残留“99”，该数字无效。实际155个GT目标中6个有可信关联、149个没有本轮定位误差（86代理、63无配对）；数值记录和汇总不受影响。图页标题的q编号是300个解码候选中的输出索引，不是900个原始Query的索引。详见 `m8c_reporting_notes.json`。
