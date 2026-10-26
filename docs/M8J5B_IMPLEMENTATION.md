# M8-J5B 实现与复现合同

基准 `6ee9bf7af7b28910def0efb9693c75790a357356`。新分支 `feat/m8j5b-coverage-aware-supervision`。新文件使用m8j5b前缀，旧源码/配置/报告/数据不变。监督与推理仍隔离，未知时间字段为null；没有Golden签核、CARLA、数据下载或训练。

## 源码和输入边界

- `datasets/m8j5b_coverage.py`：纯CPU覆盖计算与Positive诊断资格；不加载torch或模型。实例token不等于Actor；源Box几何不依赖身份匹配是否成功。
- `tools/m8j5b_build_coverage.py`：读取已存在J5A谱系和原始Semantic/Instance；输出私有逐帧监督、crop mask和K/T。
- `datasets/m8j5b_loss_adapter.py`：只有Loss路径读取Sidecar与mask，零速度权重，真实安全区域必须为空。现有真实合同若擅自设置负启用会拒绝。
- `training/m8j5b_coverage_loss.py`：官方Hungarian/Focal/L1，显式构造分类/回归权重；六层统一规则；95%Ignore和零负样本警告。未接入默认训练入口。
- `tools/m8j5b_prepare_diagnostic.py`：复用J3 RGB-only reader，仅打包images/K/T。GT供离线准备标定，不输入模型。
- `tools/m8j5b_query_diagnostic.py`：冻结模型、no_grad，前向期间文件审计钩子拒绝监督目录；只在前向后读取Loss监督；参数不变断言。
- `tools/m8j5b_visualize.py` / `tools/m8j5b_coverage_ui.html`：五帧30视图、多层开关，四个具体缺标案例。图片、标签、预测仅本地保存。

## 官方语义证据

固定CARLA源码：[ObjectLabel.h](https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/LibCarla/source/carla/rpc/ObjectLabel.h)、[Tagger.cpp](https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Game/Tagger.cpp)。Bench2Drive固定[collector](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/data_collect.py)的输入处理和save部分确认Semantic单通道、Instance原始数据。代码是SourceCodeDerived证据，不能证明压缩包捕获二进制与其完全一致；本片段实际像素通道与原始类别交叉核验，且保留分割不一致区域。

mask定义：F为两种分割的目标语义并集；S为同类源Box投影支持；D为前景范围的分割不一致；M为区域空间覆盖<0.5的完整render区域；U为未穷尽类。局部Ignore为 `dilate((F\S) ∪ D ∪ M ∪ U,4px)`。所有单像素保留，4px不是物理置信区间。跨相机任何Query完整包络投影矩形与mask相交即veto；图像外/近面不明也拒绝负。类之外的未认证空间仍受全帧退化保护。

frame_negative_enabled只有已有完整3D负证书且经过范围核验时才可为true。合成证书仅用于单测；真实片段全部false。证书字段校验不等于可信身份认证，调用方必须保留真实证据。当前类不认证自动签核或安全驾驶。

## 实际执行命令

在仓库根目录，先运行CPU构建，再执行现有隔离环境的有限测试。脚本使用独占创建，不覆盖既有产物。命令重放需单独checkout与新的私有输出根目录（可设置 `HS_M8J5B_ROOT`），不要删除归档产物。公开报告也采用独占创建，因此完整重放须在独立工作副本中使用空的本阶段报告输出目录，保留官方源码缓存与历史只读输入。

```bash
.venv-gpu/bin/python -m tools.m8j5b_build_coverage
# 实际首次在最终一致率断言失败，214帧已写完；保留该日志后：
.venv-gpu/bin/python -m tools.m8j5b_build_coverage --summarize-existing
.venv/bin/python -m tools.m8j5b_run_isolated tests --suffix final
.venv-gpu/bin/python -m tools.m8j5b_prepare_diagnostic
# 已发生一次矩阵校验失败；保留masks并创建masks_v2，未重算像素mask：
.venv/bin/python -m tools.m8j5b_align_calibration
.venv/bin/python -m tools.m8j5b_run_isolated diagnostic --suffix run2
.venv-gpu/bin/python -m tools.m8j5b_visualize
.venv/bin/python -m tools.m8j5b_validate_ui
.venv/bin/python -m pytest tests/test_m8j5b_coverage.py tests/test_m8j5a_annotation_audit.py tests/test_m8j4_gt.py tests/test_m8j4_contract.py tests/test_m8j5_supervision.py -q
```

最终源码直接按cam2ego生成masks_v2，新建复现无需align脚本；align仅为本次保留错误产物的追加修复。正式Loss只加载masks_v2。默认运行目录已存在时拒绝覆盖；isolated runner用不同suffix保留每次结果，300秒上限、禁网、无安装。已有原始数据条件下，CPU测试与synthetic CUDA tests可直接用新suffix重复验证；不必再执行模型前向。

四帧预算包含首轮失败前完成的00107前向。run2校验其缓存SHA并复用，再执行另外三帧，合计4次。00107原首轮前向耗时没有持久化，报告为null；其run2显存只是缓存Loss审计，不能当作完整前向显存。新模型调用最大allocated约1.030GiB。没有实际backward/optimizer。合成梯度测试对构造预测tensor求导，head未更新。

## 数据与测试证据

私有根：`/root/gpufree-data/hs-drive-private/m8j5b-coverage-supervision`。`sidecar`逐目标资格，`masks`原矩阵，`masks_v2`对齐矩阵，`case_reproduction.json`四例源证据，`positive_policy_difference.json`25条资格差异，`predictions_run1/run2`冻结预测与完整query状态，`web`图像开关，`cases`对照图。所有原始/衍生数据、真实逐query/对象输出不进Git。

公开报告只含聚合指标、合成测试、源码、SHA与运行日志。初始失败全部留存。历史J5的16项PETR测试与本轮15项共31通过；CPU32通过；UI9断言通过。GPU测试已检验完整Loss/梯度等价、未知0梯度、真实负梯度、正样本优先、六层辅助、全Ignore/零Positive、坏GT、极端几何、两摄遮挡否决、混合batch、未知速度、精确crop K/T和高Ignore警告。

**验收只通过工程与合成梯度，不通过真实负监督或真实训练Smoke。** 无论局部mask面积多小，都不能据此扩大负区域；后续应有独立冻结的真实负证据或类别级损失合同，再申请更新实验。
