# M8-A 补充决策：代理验证与官方 PETR 资产

**结论：ProxyVerified / AssetAccessGo。现在已从官方入口取得完整 PETR-R50-DCNv2-C5 检测 Checkpoint，并完成基本结构与隔离读取验证。PETR 整体 P0 尚未通过。**

任务依据 `478070b`，从保留 M8-A 失败交付的 `e2668bf` 建立独立分支 `feat/m8a-proxy-asset-retry`。这是代理开启后的新证据，不覆盖 M6-A、M8-A 原失败记录。本补充任务仅执行网络与资产检查，未安装 PETR 框架依赖、运行 CUDA 算子/六摄前向、开发 CARLA Adapter、训练或连接车辆控制。

## 之前的失败是否由代理网络问题造成？

**对 Google Drive 的连接失败，当前同机对照强烈支持“下载进程未使用代理”这一解释；不能据此归因所有历史百度错误。**

在 HS-Drive RTX4090 服务器发现 `clash-verge / verge-mihomo` 进程与真实混合监听端口 **127.0.0.1:7897**；活动配置为 rule 模式、TUN关闭。当前 shell 没有 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY。此端口来自进程/监听/活动配置核对，不是假定7890。未修改全局环境或代理配置。

| 同服务器测试 | 默认环境 | 显式 HTTP 代理 | 显式 SOCKS5h |
|---|---|---|---|
| GitHub 官方仓库 | HTTP200，0.88s | HTTP200，1.48s | 未另测 |
| Google 官方文件页 | curl28，约5.21s连接超时 | HTTP200，1.78s | HTTP200，1.48s |
| Python requests GitHub | 未另测 | HTTP200，1.94s | 未另测 |
| Python requests Google | 未另测 | HTTP200，6.95s | 未另测 |

HTTP 代理测试记录 CONNECT200、实际 socket 连到 loopback、TLS verification result=0；Python 使用同一端口、verify=True，取得已验证证书有效期。SOCKS5h使用代理端DNS解析。失败请求的 ssl_verify_result=0 不被单独算作TLS通过，必须同时有连接与HTTP响应。

**已验证同一本地代理入口，而非声称同一公网出口IP。** rule模式可能按域名选择不同上游；没有收集私人出口、订阅、代理节点或凭据。TUN关闭且默认下载仍失败，说明当前下载未自动受益于已启动代理；未对所有上游透明路由做穷尽证明。M6-A和原M8-A的Google超时与此失效模式一致，但历史当时所有网络条件无法追溯恢复。百度errno=2/-7属于不同服务/会话问题，本轮Google成功后未重试百度。

详见 [网络与代理审计](m8a_network_proxy_audit.json)、[同机curl记录](m8a_proxy_retry_logs/curl_probes.json)、[Python记录](m8a_proxy_retry_logs/python_probes.json)。

## 现在是否取得完整、可信的官方检测权重？

**是，达到本轮 AssetAccessGo 的来源链、完整性和基本模型合同要求。**

固定官方源码 `f7525f93467a33707ef401c587a52d5e7b34de74` 的 README 原始 R50-C5 行指向本次 Google file ID；所选配置为 `projects/configs/petr/petr_r50dcn_gridmask_c5.py`。[官方固定 README](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/README.md)。

独立新下载环境使用 gdown6.4.2、显式已验证 HTTP 代理、TLS校验、无浏览器/用户cookie、最多1次瞬态重试、30s无响应超时与900s外层总超时。一次启动约 **62.56s** 完成，最终无异常；实际内部重试数没有单独插桩，不能声称精确重试0次。[下载报告](m8a_official_checkpoint_download.json)。

文件位于：

```text
/root/gpufree-data/hs-drive-experiments/m8a-proxy-retry/assets/epoch_24.pth
435218261 bytes
SHA256: 16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad
```

文件大小与历史官方百度目录记录一致，但历史目录值不是独立哈希。文件含984个ZIP成员，全部CRC通过；静态pickle全局仅出现OrderedDict、Torch Storage与Tensor重建函数；不是HTML页面。CRC与SHA本身不证明作者身份，也不能保证无恶意内容，因此仍执行了隔离、受限反序列化。

隔离采用 bubblewrap 新网络/文件系统命名空间，清空环境、无项目目录/凭据、无NVIDIA设备、只读权重和可信运行库；独立inspect-env仅借用原已安装Torch运行库的只读挂载，**没有在主项目进程中加载checkpoint，也没有修改原环境**。使用 `torch.load(weights_only=True, map_location="cpu")`，禁止任意pickle回退；限制内存12GiB、CPU90s、墙钟120s。实测外部TCP不可达，凭据与项目目录不可见，CUDA未初始化。

最初隔离启动有一个工具错误：脚本挂载名inspect.py遮蔽了Python标准库inspect，导致Torch导入失败，尚未读取checkpoint。改为checkpoint_reader.py后重跑；原失败日志保留，未改变模型或反序列化限制。容器也不允许新挂载procfs，因此使用空proc tmpfs；这个准备失败与模型兼容性无关。

| 内容 | 实际检查结果 |
|---|---|
| DetectionCheckpoint | 558个模型Tensor，Backbone336、PETR Head222，均有限 |
| BackboneInitializationWeights | 未使用独立ResNet初始化权重冒充检测模型 |
| OptimizerState | 单独存在，含212个state条目；没有用于训练 |
| Metadata | epoch24、iter84408、10类及配置文本存在；配置只做数据读取，不执行 |
| Query / Decoder | reference_points为[900,3]；Decoder层0–5存在 |
| 分类/回归 | 六组分类输出10类、回归code size10，关键层shape匹配 |
| 几何/视觉结构 | input_proj [256,2048,1,1]、position_encoder [1024,192,1,1]、DCNv2 offset权重存在 |
| 严格完整模型加载 | **NotRun**；基本shape检查不能替代构造PETR后严格load_state_dict |

详见 [完整性与合同报告](m8a_checkpoint_integrity.json)及[全部Tensor shape](m8a_proxy_retry_logs/checkpoint_structure_v2.json)。

可信来源依据是**固定官方模型行 → 相同Google文件ID → 验证TLS的公开下载 → 完整文件及训练后检测结构一致**。未发现官方公布的预期SHA256，本地SHA只固定今后使用的同一文件，不是独立真实性证明。代码许可证为Apache-2.0，作者提供模型用于研究/评价；未核实单独checkpoint再分发或商业条款，本轮不上传权重、不主张额外权利。

## 可复核命令和源码身份

网络测试、下载和隔离检查分别冻结于 `9daeabd`、`d19a612`、`43d858c`；成功报告生成于 `7c839b8`。失败隔离源为 `1ffe7a6`。各JSON保存实际执行提交及源码SHA，不能以最终交付提交替代实验身份。

只读复核文件：

```bash
sha256sum /root/gpufree-data/hs-drive-experiments/m8a-proxy-retry/assets/epoch_24.pth
.venv/bin/python -m pytest -q tests/test_m8a_proxy.py tests/test_m8a_assets.py
```

实际下载等价命令如下，使用独立环境且不关闭TLS；此处用于复核参数，**不要求对已存在文件再次运行**。实际执行器还设置了900s外层超时并将输出脱敏。

```bash
/root/gpufree-data/hs-drive-experiments/m8a-proxy-retry/download-env/bin/gdown \
  'https://drive.google.com/file/d/1c5rgTpHA98dFKmQ9BJN0zZbSuBFT8_Bt/view' \
  --proxy http://127.0.0.1:7897 --continue --timeout 30 --retries 1 --no-cookies \
  -O /root/gpufree-data/hs-drive-experiments/m8a-proxy-retry/assets/epoch_24.pth
```

工具与协议：[代理审计](../tools/m8a_proxy_audit.py)、[下载](../tools/m8a_proxy_download.py)、[隔离启动](../tools/m8a_proxy_isolate.py)、[读取检查](../tools/m8a_proxy_inspect.py)、[协议](../configs/m8a/proxy_retry.json)。产物为排他写入，重复运行不会覆盖原失败或当前证据。下载环境包版本锁保留；隔离CPU检查的Torch2.8.0不是PETR旧依赖兼容性成绩。

## 状态及下一边界

原M8-A34个封存文件SHA核验无差异，基础已跟踪文件无修改/删除；原Python环境包版本不变。14项相关合成单测通过；它们验证工具脱敏、资产条件和形状拒绝，不是PETR CUDA测试。[保全](m8a_proxy_retry_logs/preservation.json)、[测试日志](m8a_proxy_retry_logs/tests_final.txt)。

新证据允许后续按原M8-A检查独立PETR依赖、CUDA算子、严格模型加载和RTX4090原规格前向。本补充任务到此停止，以上后续工作本轮均NotRun；不提前训练、CARLA适配、默认模型替换或视觉控制。历史AssetNoGo记录保持原值，新资产决策只适用于本次新条件与该SHA文件。

M5-O CandidateNoGo、M7 OnlinePlannerNoGo、ControlNoGo、RuntimePartial、Occupancy NotPassed、视觉SafetyNotValidated均不变。**AssetAccessGo不等于FeasibilityPassed，更不等于可靠3D感知。**
