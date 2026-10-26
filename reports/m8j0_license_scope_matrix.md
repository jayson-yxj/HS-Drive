# M8-J0 目标片段许可用途矩阵

实际检查：2026-10-26。这是官方声明、证据边界及下一步授权条件审查，不是法律意见。目标仅为 `DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz`，不扩展到其他Bench2Drive发布物。

固定GitHub提交 `7ec25d1c9f7522d923ce5f3420986cef1cb2d956` 的 [LICENSE](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/LICENSE) 是CC BY-NC-ND 4.0；[README许可段](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/README.md#license)将其声明用于资产与代码，除非另有说明。两项原文、Git blob及SHA已在来源清单核验。

[官方HF数据卡](https://huggingface.co/datasets/rethinklab/Bench2Drive/blob/main/README.md)的YAML和正文均声明Apache-2.0，不能将差异归结为“只有标签写错”。浏览服务读取的[README修订](https://huggingface.co/datasets/rethinklab/Bench2Drive/commit/ce56cfbc26edf7796586e26fd44644f0ac2f6bcc)日期为2024-08-14；GitHub LICENSE历史显示2024-11-02提交 `fd3009a16f551c6d36139c831a4a172f17b74419` 更改许可。这支持“声明可能未同步”的假设，**不证明目标文件被重新授权、旧授权失效或HF构成资产例外**。目标文件本身修订与附带条款尚未验证。

| 用途 | 状态 | 条件与当前证据边界 |
|---|---|---|
| 下载与本地存储 | `LicenseScopePartial` | 非商业内部复制在CC许可2(a)(1)中有依据；Apache也通常允许复制。尚未闭合目标文件适用声明、附加条款及官方访问。不是断言下载违法，也不是本轮下载授权。 |
| 非商业学术研究 | `LicenseScopeSupported`（声明层，条件成立时） | 官方项目公开科研用途及两种声明均给出研究可用基础；CC路径须符合非商业、署名等适用条件。本项目实际用途性质不能由“个人开发”自动推定。资产级适用范围仍Partial。 |
| 在公开数据上训练研究模型 | `LicenseScopePartial` | CC 2(a)(1)(b)允许非商业制作、复制但不分享改编材料，不能把ND理解为禁止一切内部训练；“公开可访问”也不消除许可。训练、模型法律属性及后续发布分开核查，本轮均不执行。 |
| 生成及发布评价图表 | `LicenseScopePartial` | 不包含可识别源图像的聚合数值图表与图像裁剪、叠加GT、拼图不等同；前者通常不直接再分发原素材，后者可能涉及改编或分享。不能把统一“图表”许可用于两类产物。需核实图像示例发布范围并保留来源署名。 |
| 发布衍生标注 | `UseRequiresAdditionalPermission`（若属于受限改编/数据库分享） | 未证明标签可与受保护内容独立；CC ND分享限制与Apache声明差异未消除。先取得目标资产明确条件或确认独立适用例外，不在本轮发布。 |
| 公开训练权重 | `LicenseScopePartial` | 权重是否构成受许可材料的改编/再现未取得资产级说明；不自动把训练权重视为数据再分发，也不凭HF标记批准商业或公开发布。 |
| 再分发原始数据 | `LicenseScopePartial` | 若CC确实适用，非商业原样分享在署名等条件下有依据；HF Apache路径的通知义务不同。单个压缩包内部还可能有第三方条件，当前未核验，不进行镜像。 |
| 再分发处理后数据 | `UseRequiresAdditionalPermission`（受限改编时） | 技术格式转换不必然形成改编，不能一概禁止；裁剪、重标注、组合等需要逐项判断。授权冲突未解决时不批准对外分享。 |

全资产综合状态：`LicenseScopePartial / LicenseConflictUnresolved`。具体用途条件不能互相替代。CC法典中的非商业复制、非公开改编及纯技术修改条款有区别；Apache允许复制和改编但分发仍涉及许可副本、变更说明、NOTICE等义务。[CC官方说明](https://creativecommons.org/licenses/by-nc-nd/4.0/)、[Apache官方条款](https://www.apache.org/licenses/LICENSE-2.0)。

固定GitHub树中检出顶层、leaderboard、scenario_runner及vendor各自LICENSE路径，没有找到目标片段专用声明；HF完整目录和压缩包内部未取得，不能写“没有额外条款”。CARLA 0.9.15固定[README](https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/README.md#licenses)区分代码MIT、CARLA资产CC-BY及Unreal独立条款。这不证明所有第三方对象、纹理或渲染产物已统一清权，也不意味着每幅输出图片自动继承引擎许可。

有界维护者说明检索：GitHub仓库限定关键词`license`返回0项，HF讨论页当前浏览快照没有帖子；没有找到解决该片段许可差异的维护者解释。不是全互联网不存在说明的证明。没有向任何人发送消息。

可发送的简短询问草稿（**DraftOnly / NotSent**）：

> We are reviewing only `rethinklab/Bench2Drive/DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz` (280176944 bytes; SHA256 b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2), listed at GitHub commit 7ec25d1c9f7522d923ce5f3420986cef1cb2d956. The GitHub license/README state CC BY-NC-ND 4.0, while the HF card states Apache-2.0. Which license and exact HF revision apply to this archive, and are there extra asset/third-party terms? Our immediate proposal is one noncommercial local download and a private one-frame RGB/calibration/annotation review, with no redistribution or training. Separately, could you clarify noncommercial research training, publishing aggregate metrics or annotated images, derivative annotations, trained weights, and raw/processed-data redistribution? A versioned asset-specific statement would help us preserve the correct attribution and conditions.

在许可适用范围与服务器访问证据闭合前，不提出实际下载授权申请；无需以CARLA重启或新数据源搜索代替这个缺口。
