# M8-J1 许可澄清结果

检查日期：2026-10-26。**LicenseConflictUnresolved / LicenseScopePartial**。这是官方证据和使用条件核查，不作法律意见；没有取得指定旧Mini片段的一致资产级官方说明。

| 声明层 | 本轮证据 | 适用边界 |
|---|---|---|
| 固定GitHub LICENSE | 服务器API200，解码原文19133bytes，与历史冻结SHA `0bb835f5ed7272947a23d92d0e74bbdadcbf0beb9235c0735c32d1310a4bceda`一致；CC BY-NC-ND4.0 | 固定仓库版本的许可声明 |
| 固定GitHub README | API200，25070bytes，与历史SHA `09e725b10cf17ee1736a28ca1a0ba8ba0344594d7064e5ca47906f62044bc368`一致；资产和代码适用CC，另有说明除外 | 没有发现目标片段明确例外 |
| HF数据卡 | 独立浏览服务可读缓存页，YAML及正文均Apache-2.0 | 不是GPU服务器取回的固定对象许可；不能当作冲突已消除 |
| 指定旧Mini压缩包 | 未下载，未验证内部LICENSE/NOTICE；未取得目标对象修订与LFS/Xet身份 | `PayloadNotVerified`；额外条款并非已证明不存在 |

来源：[固定LICENSE](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/LICENSE)、[固定README](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/README.md)、[官方HF卡](https://huggingface.co/datasets/rethinklab/Bench2Drive/blob/main/README.md)。原始API及浏览结果保存在 `m8j1_sources`。

M8-J0记录的HF卡修订日期早于GitHub许可变更，仅是声明可能不同步的线索。没有由此推断旧授权被撤销、目标文件双重许可，或自动选择较宽松Apache条款。

本轮有界GitHub issue搜索 `repo:Thinklab-SJTU/Bench2Drive license OR licensing OR permission OR Apache` 返回2项：#215云服务器启动、#184 checkpoint读取；命中的是系统报错中的permission，不是资产授权说明。结果未截断。HF讨论页缓存没有显示帖子；这不是所有渠道都不存在解释的证明。未取得维护者对目标片段的正式澄清，未对外发出问题。[询问记录及最终草稿](m8j1_maintainer_contact_record.md)。

| 本轮限定用途 | 当前判断 |
|---|---|
| 非商业内部下载与单帧审核 | 条件性支持但资产范围未闭合。CC2(a)(1)对非商业复制有依据，不能声称禁止全部内部研究；访问及附加条款仍未知。 |
| 非商业研究模型训练 | Partial。内部训练与公开权重/数据分享分开；不能把ND理解为一律禁止内部制作，也不能以“公开数据”免除使用条件。 |
| 聚合指标及少量标注图像发布 | Partial。纯统计指标和含源图像的标注图不等同，后者的分享/改编条件需要明确。 |
| 模型权重或衍生标注公开 | Unresolved。未取得权重属性、标签分享及相关通知义务的资产级解释，不自动准许或一概禁止。 |
| 原始或处理后数据再分发 | Unresolved。原样非商业分享与处理后改编分享条件不同；不能统一沿用HF标签。 |

第三方条件沿用已核验CARLA0.9.15声明层边界：代码MIT、CARLA自身资产CC-BY、Unreal独立条款；目标包实际资产来源/NOTICE没有证据，不能据此判断渲染输出全部清权或自动继承引擎许可。[原官方CARLA说明](https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/README.md#licenses)。

本轮没有实质许可澄清进展，选择B所要求的“一次有界核实”已经完成；并未声称完成维护者沟通。后续如有真实资产说明，可按限定用途重审；目前不完善下载申请，不追加数据工程。
