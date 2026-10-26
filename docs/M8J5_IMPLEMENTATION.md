# M8-J5 实现与人工审核流程

2026-10-26；基线 f8e9f9c。所有新实现为 opt-in，未改 PETR 运行时、默认模型/控制及历史 J3/J4 文件。单人审核和真实模型更新仍是独立门禁。

## 代码

- datasets/m8j5_supervision.py：独立可信轴、Box 中心/尺寸/Yaw 转换及签核校验。
- datasets/m8j5_loss_adapter.py：审核记录到监督对象；不向 RGB packet 注入 GT，不自动生成安全负证书。
- training/m8j5_ignore_loss.py：官方 Head 的独立 Loss 包装，匹配可靠正目标，按空间审核状态筛选负监督，六层规则一致。
- tools/m8j5_build_review.py：214 组可信侧车与 36 个候选，所有文件在私有目录。
- tools/m8j5_build_review_ui.py / m8j5_review_ui.html：每对象独立六摄标注、嵌入图像的离线交互页。
- tools/m8j5_review_gate.py：只校验实际人工交付的记录，不能生成批准或验证身份真实性。
- tools/m8j5_run_isolated.py：既有 PETR Python3.8 环境中运行有限测试/前向，禁网络、不安装依赖，240s 超时退出。
- tools/m8j5_forward_regression.py：一次冻结六摄前向及无签核的 Query 策略，零真实反向更新。

Supervision.boxes 使用原 PETR gravity center xyz,w,l,h,CW yaw,vx,vy。HS 中心无需下降半高；只有构造框架 LiDARInstance3DBoxes（底中心定义）时才减 h/2。回归编码的 XYZ 为米，尺寸取 log，Yaw 为 sin/cos。未知速度占位为 0，但必须将对应损失权重设为 0，不得监督“速度为零”。真实标签只映射 car=0、pedestrian=8，其余保持 Unknown。三态不互相覆盖几何可信维度。

negative_certificate 是监督路径中的审核声明，不是网络输入。它需要 Verified、所有十类 class_ids、reviewer、evidence_ref 与 scope（BoundedRegion 或完整标注），不接受仅车/人证据认证所有类别不存在。当前 adapter 固定没有安全负区域，禁止从光线/Instance 背景推导 Free。未知或不可靠 Box 不能进入匹配回归；可解释几何的 Unknown extent 可用于 veto。若几何也未知，不能假设其在任何区间之外，必须限制负区域认证。

Loss 的空间策略使用 detach 后的预测完整包络。边界包含/相交规则和 0.5m margin 固定；不是连续碰撞检测，也不是物理 Free 证明。安全负区域不能仅依据预测置信度生成。Positive 优先；所有非 Positive 回归零权重；所有辅助 Decoder 独立匹配、执行同样规则。完整标注模式已与官方原 Loss 和梯度比较；两阶段、DDP 与混合精度未验收。

## 已执行命令

在仓库根目录，复用本机已有环境；生成器拒绝覆盖文件。失败日志和运行后继版本分别保留，不反复覆盖原输出。

```bash
.venv-gpu/bin/python -m tools.m8j5_build_review
.venv/bin/python -m tools.m8j5_run_isolated tests --suffix release
.venv/bin/python -m pytest tests/test_m8j5_supervision.py tests/test_m8j4_gt.py tests/test_m8j4_contract.py tests/test_m8j3_rgb_clip.py -q
.venv/bin/python -m tools.m8j5_run_isolated forward --suffix run2
.venv-gpu/bin/python -m tools.m8j5_build_review_ui
.venv/bin/python -m tools.m8j5_test_review_ui --suffix run2
.venv/bin/python -m pytest tests/test_m8j5_supervision.py -q
```

重跑 tests 可给新的 suffix；真实前向只在确有回归需求时重跑，不是训练。构建数据/页面只允许在新私有根目录或保留原产物后的独立复现目录执行，不删除本轮原始记录。protocol 中 private_root 控制私有目录，source_sidecar_root 继续指向已存在 J4；没有新下载。release manifest 绑定本次代码/日志 SHA，原官方源码 SHA 单独冻结。

## 审核页面

打开 /root/gpufree-data/hs-drive-private/m8j5-supervision/review_index.html，或把该 HTML（约 21.9MB）复制到本机浏览器。页面本身包含所有图像，无需访问服务器；CSP 禁止 connect，未使用 fetch、WebSocket、localStorage 或后台存储。旧静态页保留在 review_index_static.html。

1. 本人填写真实审核者与证据来源。不得以“自动审核”代替真实身份签核。
2. 看当前候选的六摄 RGB 投影、Instance 对照、几何/类别/身份可信状态和 Box 八角点。图片可点击放大。每次只高亮一个候选。
3. 选择“通过／拒绝／无法确定”。通过需四项确认；范围外、地下或超出 yaw-only 表达的候选不能通过。选择“无法确定”是有效审核结果，但不是 Golden。
4. 修改确认项或备注会撤回该项旧决定，须重新点击操作记录时间。刷新或关闭会丢失页面内存中的决定；没有自动保存。
5. 36 项全部完成后，页面自动形成模板 JSON，点击“导出审核 JSON”下载 m8j5_review_responses.json。时间为浏览器实际点击时间，明确标注未平台验证；未补造原数据时间戳。
6. 由用户将导出文件保存/复制至服务器 /root/gpufree-data/hs-drive-private/m8j5-supervision/m8j5_review_responses.json，并提供真实提交来源。浏览器不能直接写入远程目录。

校验命令（只有真实文件已提交后执行）：

```bash
.venv/bin/python -m tools.m8j5_review_gate \
  --responses /root/gpufree-data/hs-drive-private/m8j5-supervision/m8j5_review_responses.json \
  --human-source-reference '填写实际审核者提交记录或本次对话说明；不编造消息ID' \
  --output /root/gpufree-data/hs-drive-private/m8j5-supervision/human_review_validation.json
```

校验器检查候选内容 SHA 与对象字段未改、四项确认及合法决定、时区时间、实际来源；结构通过不等于验证人的身份。不要使用 UI 自动测试输出充当人工输入；所有浏览器测试均明确为 SYNTHETIC，未生成真实审核结果。

页面三种决定直接映射 ApprovePositive、Reject、NeedsReview。所有返回字段与现有模板一致。Golden GT 候选的私有 SHA 不含页面内嵌图片字段；导出的 SHA 绑定原始候选对象、可信状态及选择理由。当前已签核数为 0。

## 下一步门禁

真实 Smoke 仅在真实签核、坐标合同、Loss 测试和监督来源均通过后申请执行，预算预声明为最多 2 帧/3 步、8GiB PyTorch allocated、180 秒。当前没有实现自动训练入口，也没有调用 optimizer，避免未签核数据误启动。即使仅做正样本更新，也需明确它只能验证链路，不能证明检测训练质量。M8-J6 小规模检测训练还需可靠负区域或明确的监督限制、合理的预声明预算和真实 Smoke 验收；不能用这一片段随机拆分宣称泛化。

当前状态：Loss 工程通过；人工 Golden Pending；真实更新 NotRun；正式训练 No-Go。
