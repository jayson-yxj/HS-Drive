# M8-J5C 实施与复现

基础4143011，独立分支 `feat/m8j5c-numerical-supervision-closure`。仅新增文件，默认模型、Loss、Planner、原始数据、历史失败报告不变。

新增文件职责：

- `configs/m8j5c_protocol.json`：数值容差、原/原及新/新及原/新、固定8帧和任务范围，在实验之前写入并保留SHA。
- `tools/m8j5c_numeric.py`：复用J5BV fixture与官方Loss；恢复完整状态和RNG，记录全部参数梯度、模块边界和三组误差。CPU侧浮点64只用于误差统计，模型仍使用报告中的精度。
- `tools/m8j5c_dcn_minimal.py`：随机合成输入与相同上游梯度的独立官方DCNv2复现。
- `datasets/m8j5c_detection_scope.py`：独立候选任务资格，Identity Unknown不等于几何无效。未授予任何训练权限。
- `tools/m8j5c_candidates.py`：只读复用214份元数据，仅读取固定8帧图像；原始与排除对象均可追溯。
- `tools/m8j5c_review_ui.py`：私有六摄对照、27项具体疑点、17项浏览器检查，不生成签名。
- `configs/m8j5c_smoke_proposal.json`：明确而被阻止的未来Smoke配置；preflight只读且返回非零，不加载模型。

## 数值最小命令

```bash
.venv/bin/python -m tools.m8j5c_run_numeric --name local_numeric --legacy-determinism
.venv/bin/python -m tools.m8j5c_run_numeric --name local_dcn --legacy-determinism --minimal-dcn
.venv/bin/python -m pytest --noconftest tests/test_m8j5c_scope.py tests/test_m8j5bv_eligibility.py -q
.venv/bin/python -m tools.m8j5c_smoke_preflight
```

最后一条预期退出2（NoGo）。Runner使用原隔离环境、禁网、上限600秒；每个name只能创建一次，避免覆盖。最小算子不需要私有输入或权重。严格Torch确定性诊断不加 `--legacy-determinism`，预期重现原版索引错误。

数值运行保存软件版本、状态SHA、随机状态SHA、dtype、train/eval、BN模式、Scaler不适用或手工缩放、TF32、确定性设置、全部误差及异常张量。所有合成梯度张量只存私有实验目录，不提交模型。官方CUDA源码路径、相关行及SHA在 `reports/m8j5c_sources/installed_operator_evidence.json`。

## 候选材料

`tools.m8j5c_candidates` 拒绝覆盖；本轮已执行，不需要重新扫描。开发复现使用下列新目录环境变量（每个目录只可运行一次），不能删除历史输出凑重跑。核心范围测试无需私有数据。

```bash
export M8J5C_CANDIDATE_ROOT=/root/gpufree-data/hs-drive-private/m8j5c-candidate-next
export M8J5C_CANDIDATE_REPORT="$M8J5C_CANDIDATE_ROOT/reports"
.venv/bin/python -m tools.m8j5c_candidates
.venv/bin/python -m tools.m8j5c_review_ui
```

私有页面：
`/root/gpufree-data/hs-drive-private/m8j5c-numerical-supervision/candidates/web/index.html`

源图像和Semantic/Instance通过本机符号链接读取；转换Box、mask、逐对象问题均保持私有。截图经助手观察只证明页面显示，不能冒充人类覆盖签核。全量214帧原因表也只存私有目录，公开报告保留分类统计。

## 被阻止的具体Smoke方案

候选00076、00107，batch1、固定顺序00076→00107→00076，最多3步，FP32、不使用AMP/Scaler、Backbone冻结eval、既有Head训练、AdamW lr1e-5、weight_decay0.01、gradclip35、无梯度累积、无随机增强、原512×1408crop及标定不变。限制180秒/22GiB；资源或数值异常立即停止。真实权重另存私有目录。

这**不是可执行授权**：两帧覆盖仍未闭合，car/person分类项与其他类别屏蔽的专用Loss及范围外Ignore尚需验证，真实全分辨率反向显存亦未测。先取得真实六摄定向审核、未解释前景处理依据、有效负监督及P/N权重统计，再进行一次无更新的全分辨率资源检查；最后仍需独立批准真实更新。只要任一项缺失，preflight保留NoGo。当前本片段退出为B，不自动推动这些前置任务。

## 证据边界

历史4帧×6层的900Query统计仅引用旧报告，未伪装成新的两类任务训练统计。本轮30项CPU检查、17项实际浏览器检查；数值是GPU研究矩阵，不将预期AMP失败算成“全部测试通过”。FP32结果限定单seed、小合成输入、当前环境，不能外推真实训练或安全驾驶。
