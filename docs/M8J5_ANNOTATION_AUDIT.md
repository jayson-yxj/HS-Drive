# M8-J5 追加审计实现与复现

新分支 `feat/m8j5-raw-annotation-audit`，基于已完成 M8-J5 分支。新文件统一 `m8j5a_`，保留旧报告/源码/Sidecar；不修改默认模型和训练入口。该审计取代要求用户立即签核 36 个候选的工作重点。

入口与产物：

- `datasets/m8j5a_annotation_audit.py`：全类别监督几何，几何保留不依赖实例身份；Box 中心、半尺寸及逆变换验证；灯/标志代理单独标记。
- `tools/m8j5a_audit.py`：214 帧全部对象逐条追踪，所有六摄投影及 render 区域与源 Box 空间关系；保留 source ordinal，真实时间字段仍为 null。
- `tools/m8j5a_cases.py`：12 例原选择性展示 / 全原始 GT / Instance 对照。
- `tools/m8j5a_finalize_audit.py`：独立官方针孔函数与 cam2ego 投影对照，生成逐帧/相机/类别及谱系汇总。
- `tools/m8j5a_build_ui.py`、`tools/m8j5a_audit_ui.html`：214 帧全对象三层显示，没有签核动作。
- `tools/m8j5a_validate_ui.py`：真实私有页禁网浏览器回归。
- `tests/test_m8j5a_annotation_audit.py`：半尺寸、中心、静态异常坐标、未知身份、投影、实际载荷与具体缺框回归。

本次已执行命令（仓库根目录）：

```bash
.venv-gpu/bin/python -m tools.m8j5a_audit
.venv-gpu/bin/python -m tools.m8j5a_cases
.venv/bin/python -m tools.m8j5a_build_ui
.venv/bin/python -m tools.m8j5a_validate_ui
.venv/bin/python -m tools.m8j5a_finalize_audit
.venv/bin/python -m pytest tests/test_m8j5a_annotation_audit.py tests/test_m8j4_gt.py tests/test_m8j4_contract.py tests/test_m8j5_supervision.py -q
```

`.venv-gpu` 仅为已有含图像库的 Python 环境，本轮脚本不加载网络、不调用 CUDA、不训练。全载荷主体审计约 77.09 秒。Python 回归 24 passed、0 skipped；UI 12 断言通过。日志位于 `reports/m8j5a_sources`；包含逐目标的日志移至私有目录。

审计生成器使用独占创建，防止覆盖已有证据。首次页面构建时已归档旧入口；最终代码将此动作改为显式 `--update-review-entry`，默认复现不触碰旧页面。已有归档时该选项会拒绝重复修改。最终脚本支持独立输出目录，已实际完成第二份完整复现，结果见 `m8j5a_reproduction_validation.json`。复现时为每次运行选择一个尚不存在的私有目录，例如：

```bash
export HS_M8J5A_OUT="/root/gpufree-data/hs-drive-private/m8j5a-replay-$(date -u +%Y%m%dT%H%M%SZ)"
export HS_M8J5A_REPORTS="$HS_M8J5A_OUT/reports"
.venv-gpu/bin/python -m tools.m8j5a_audit
.venv-gpu/bin/python -m tools.m8j5a_cases
.venv/bin/python -m tools.m8j5a_build_ui
.venv/bin/python -m tools.m8j5a_validate_ui
.venv/bin/python -m tools.m8j5a_finalize_audit
```

保持原始输入和 OLD 只读，不删除已有输出。全局报告目录不会因上面命令被覆盖；文件已存在会失败。复现只自动生成诊断图，不自动生成“已目视审核”声明；本次实际图像检查另有追加记录，不是人工签核。

私有根目录 `/root/gpufree-data/hs-drive-private/m8j5a-raw-audit`：

| 路径 | 内容 |
|---|---|
| `frames/*.json` | 全原始对象、转换后角点/中心、旧 Sidecar、每个筛选原因、六摄投影 |
| `lineage_all_records.json` | 3741 条非自车记录完整谱系与非互斥原因 |
| `render_regions.json` | 包括小于100px区域的完整语义/render分组；绝不作为 Actor 身份 |
| `per_frame_camera_class.json` | 按文件序号、相机、类别计数 |
| `raw_annotation_hashes.json` | 214 个原始 annotation SHA；执行后重新核对 |
| `cases/` | 12 例全分辨率对照与 contact sheet；AI目视记录不是人工签核 |
| `web/index.html` | 新完整审计页，本地图像符号链接，0网络连接 |

全对象链路逐项区分：raw record retained、geometry valid、view projectable、identity evidence supported、historical positive candidate、PETR technical candidate、final training enabled。`project_box.status=Invalid` 是“本视图投影无效”，不能解释为 3D Box 无效。`NoSupportedViewInPETRCrop` 同时依赖旧关联资格，不是纯几何裁剪率。所有原因同时保存，因而不可将各项相加解释为互斥丢失数。

区域空间支持定义：在同类原始 Box 的投影凸包内，覆盖某个语义/render区域至少50%的像素。该比例只帮助定位未解释区域，不是身份认证、可见性真值或安全背景。≥100px仅用于报告/可视化优先级，小区域从未转换成负样本。跨帧 render token 可能复用；相邻帧与六摄观测强相关。

几何模型域诊断只回答是否满足旧范围/裁剪/倾角；不放开监督。全部训练为 NotRun。原始标签没有补写/移除，缺失行人不插值，背景无框不自动负监督；所有未来训练仍须验证 Positive/Ignore/Safe Negative 的独立合同。

源代码记录使用最终文件 SHA、基准 commit 与日志 SHA，不伪造未记录的早期运行 commit。主体审计后增加了 center/extent防御检查；最终 24 项回归重新在全部214帧调用该检查，输出几何与旧 Sidecar逐项一致。独立投影汇总为最后源码执行结果。发布 commit 可从包含 `reports/m8j5a_release_manifest.json` 的 Git 提交取得，清单不循环引用自身。
