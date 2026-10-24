# M8-A 实现、实际运行和复现

基线：`478070b51f03daf7e4ff2e7cb0b0a96e7507d9ba`。分支：`feat/m8a-petr-p0-feasibility`。实际终止于 **AssetNoGo**；不存在已验证的 PETR 安装或推理命令。

## 实际执行链

1. 完整阅读本轮任务、M8-0决策及M6-A失败记录；只读核对官方源码提交、README/config/LICENSE SHA及模型行。
2. 编写资产取得协议、追加式记录工具及门禁测试，在 `242487ef96038b67670ed3f720c39c892df3e136` 冻结。
3. 保存现有环境版本清单；使用标准库和系统curl，无新依赖、无项目凭据。按协议执行一个Google下载请求和一个百度目录流程；失败后结束。
4. 在 `e0bfa7d` 冻结失败报告生成器，执行保全和报告生成；后续实验全标NotRun。
5. 最终11项工具单测通过；交付检查复核JSON门禁、链接、来源SHA和历史文件保护。

## 文件与安全边界

| 文件 | 职责 |
|---|---|
| [protocol.json](../configs/m8a/protocol.json) | 唯一模型、官方入口、无重试/超时/容量上限、下游依赖、六摄原规格及禁止项 |
| [m8a_common.py](../tools/m8a_common.py) | SHA、实际Git提交、协议身份和排他写入，不导入CARLA/训练代码 |
| [m8a_assets.py](../tools/m8a_assets.py) | 官方公开入口访问、无持久化cookie的百度会话、有限日志和资产候选判别 |
| [m8a_report.py](../tools/m8a_report.py) | 只接受真实AssetNoGo输入的报告生成器；保全检查；不会触发pip/GPU |
| [test_m8a_assets.py](../tests/test_m8a_assets.py) | HTML伪权重、完整条件、访问限制和追加式证据测试 |

下载到类似ZIP/Pickle的文件只叫候选，不能自动通过。资产准入要求来源、适用使用范围、完整字节、SHA、隔离可读、检测Head存在、关键shape匹配全部满足。本次后二级代码没有执行，且没有开发未经触发的完整模型部署/Adapter。

网络日志保存URL的稳定来源部分、返回码、耗时、响应字节数和SHA、过滤后的目录条目。临时cookie、签名下载URL及会话token不进入报告；因此没有保留带登录会话的原始完整HTML。原始网络异常文本和业务errno保留；响应SHA不能替代未下载模型的SHA。

本轮实际机器为RTX4090 24564MiB、驱动580.126.09；既有Python3.10.12环境分别含CARLA0.9.15及Torch2.8.0。它们只是保全清单，**不是新的PETR依赖锁**。没有安装新的wheel，MMCV/DCNv2/CUDA架构没有本轮实测。计划的独立目录 `/root/gpufree-data/hs-drive-experiments/m8a-petr` 未因资产失败而创建部署环境。

## 实际命令

下列资产命令已经执行；重复执行同一run-id会因目录已存在而失败，以保护证据。它不是要求再次联网重试。再次资产尝试应先重新明确授权与预算。

```bash
.venv/bin/python -m tools.m8a_assets --run-id official_attempt_01
.venv/bin/python -m tools.m8a_report --attempt reports/m8a_execution_logs/official_attempt_01
```

两命令分别使用上面列出的冻结提交；现有报告同样拒绝覆盖。报告输入已保留，清理或删除旧报告后重跑不是本项目推荐复现方式。

可直接执行的只读检查：

```bash
.venv/bin/python -m pytest -q tests/test_m8a_assets.py
.venv/bin/python - <<'PY'
import json
from pathlib import Path
acceptance = json.loads(Path('reports/m8a_acceptance.json').read_text())
assert acceptance['overall'] == 'AssetNoGo'
assert acceptance['A_AssetAccess']['complete_trained_checkpoint_obtained'] is False
for gate in ['B_Dependency', 'C_ModelLoad', 'D_Forward', 'E_Resource', 'F_Contract']:
    assert acceptance[gate]['status'] == 'NotRun', gate
assets = json.loads(Path('reports/m8a_official_assets.json').read_text())
assert assets['checkpoint_sha256'] is None and assets['actual_checkpoint_bytes'] == 0
print('Observed AssetNoGo; five downstream gates NotRun; no checkpoint bytes.')
PY
```

为避免误执行，没有提供“假定权重存在”的安装/训练/推理流水线。依赖安装、算子测试、模型构造、3+20次前向、显存曲线、输入输出文件、投影闭合测试均为条件NotRun。

## 日志、身份与保全

实际请求日志位于 [official_attempt_01](../reports/m8a_execution_logs/official_attempt_01/outcome.json)，每个event独立JSON。冻结文件记录协议及执行代码SHA、官方源文件大小/SHA和对应模型行；历史目录文件大小不做实际模型大小。

全部九份JSON有生成提交与源码SHA；具体访问提交另由 `acquisition_git_head` 或 `acquisition_source_commit` 指明。报告生成提交与网络执行提交不能混称。最终文档/结果提交由Git记录，交付manifest自身由最终Git提交保护，不使用自引用SHA。

保全验证覆盖基线全部已跟踪文件不被修改/删除、M7-E525项清单、两个代表模型、历史关键manifest和原环境版本。未做全外部资产重哈希；未启动CARLA检查Physics，但代码变更中无任何控制/Physics文件。

交付：[验收](../reports/m8a_acceptance.json)、[资产](../reports/m8a_official_assets.json)、[checkpoint](../reports/m8a_checkpoint_validation.json)、[依赖](../reports/m8a_dependency_matrix.json)、[算子](../reports/m8a_cuda_operator_tests.json)、[加载](../reports/m8a_model_load.json)、[前向](../reports/m8a_4090_forward.json)、[资源](../reports/m8a_gpu_memory_latency.json)、[投影合同](../reports/m8a_camera_projection_contract.json)、[总结](../reports/m8a_summary.md)、[风险](../reports/m8a_risk_and_exit_decision.md)。
