# 路线 C：Bench2Drive 已有数据合同

2026-10-26，只读检查官方代码/文档/文件清单/网页，没有下载任何数据包。固定 GitHub `0.0.4` commit `7ec25d1c9f7522d923ce5f3420986cef1cb2d956`，与历史 M7-A 查询到的提交相同；此次仍重新检查，不能把旧审查自动当新数据验证。

## 版本、大小与取得条件

| 发布物 | 官方声明 | 本轮可核验证据 |
|---|---|---|
| 旧 Mini | 10 clips，README 约4G | 官方清单总计 **2,812,733,742 bytes**，约2.62GiB；单段99,078,817–487,312,838 bytes；含逐包SHA256，但包未取得 |
| 旧 Base | 1000 clips、约400G | HF卡显示约335GB；网页数字与README约数不混同实际下载测量 |
| 旧 Full/Sup | README 合计13638 clips、约4T | 不在本轮下载或候选预算内 |
| 0.0.4 主数据 | 44类×25路=1100；含新增Occupancy；不含Depth | README约400G、HF页面约415GB；未取得实例样本 |
| 0.0.4 Depth | 独立包，改用NPZ避免旧PNG精度损失；README约7.3T | 旧HF仓库已迁移，页面仅剩5.4kB元数据；不是实际Depth体积 |

依据：[官方固定 README](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/README.md)、[Mini 清单及SHA](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/bench2drive_mini_10.json)、[0.0.4更新](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/v004_update.md)、[旧Base官方HF卡](https://huggingface.co/datasets/rethinklab/Bench2Drive)、[新版卡](https://huggingface.co/datasets/rethinklab/Bench2Drive-V0.0.4)。新Depth入口由官方卡指向 `Bench2DriveData/Bench2Drive-V0.0.4-depth`，本轮不访问载荷，不把旧入口5.4kB误算为可下载数据规模。

Mini 是旧训练数据的子集；Dev10 是闭环路线开发集，**不是 Mini 数据包的别名**。0.0.4新增220验证路线定义不等于已经取得220路标注验证数据。现有官方 Train/Validation 名称也不能替代 HS 的位置隔离、Calibration、新Final；历史看过的路线、Town近邻和相机视角不能重算独立场景。

## RGB及标定

固定 [data_collect.py](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/data_collect.py) 的 `sensors()` 明确六个环视RGB，另外还有top-down视图；**top-down不进入六摄RGB输入**。六摄1600×900，pitch/roll均0，源CARLA轴 x前/y右/z上。源值由 AST literal 提取保存在证据JSON，不执行代码。

| Camera | B2D位置xyz(m) | B2D yaw/FOV(°) | HS当前yaw/FOV(°) |
|---|---|---|---|
| Front | 0.80,0,1.60 | 0/70 | 0/100 |
| Front left | 0.27,-0.55,1.60 | -55/70 | -60/100 |
| Front right | 0.27,0.55,1.60 | 55/70 | 60/100 |
| Back | -2,0,1.60 | 180/110 | 180/100 |
| Back left | -0.32,-0.55,1.60 | -110/70 | -120/100 |
| Back right | -0.32,0.55,1.60 | 110/70 | 120/100 |

HS `configs/cameras.yaml` 前摄pitch=-5°、其余=-7°，采集profile960×576、现有输入640×384。不能复制HS K/T，也不能把B2D缩放后假装原HS布局。应读每帧实际intrinsic、cam2ego/world2cam，明确光学轴转换、resize/crop仿射与相机名称映射，单独测试角点投影。M8-B Adapter 有固定图像shape/相机角度白名单，不是任意六摄都可直接通过；只能复用隔离设计和数学函数，未来另建显式版本adapter。

源RGB JPEG quality20，有明显有损压缩风险；须诊断远场小目标支持，不能把仅尺寸更大解释为更高有效画质。固定数据采集源码仍保存灰度Depth PNG，而0.0.4更新文档声明NPZ；这是发布物/源码版本对应需样本核验的差异，不推断新版包仍是旧PNG。

## 标注可评价边界

[官方 anno.md](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/anno.md) 列出动态/静态Vehicle、Walker、交通灯/标志；含ID、type_id、源Box center/extent、world_cord、各对象变换、部分速度及road/lane/section。extent是半尺寸，不能误作完整l/w/h。Depth/Instance/Semantic与RGB声明同安装；逐摄可见像素、遮挡和匿名背景的可评价Mask仍须独立推导与审查。Cone/Barrier专门完整身份、全背景完整对象真值、精确Mesh、轮胎接触均 **NotAvailable**，不能从交通标志trigger推导障碍物实体。

文档明确行人speed可能全0，需逐帧位姿和可靠时间差重算，不能把0当静止证据。静态车location/rotation存在已知问题；文档文字对center坐标域亦有不一致：风险说明称local，Schema注释称world。源码构造center使用actor transform，且从level boxes到static actor有最近XY匹配（最多20m）再消耗候选Box；**不是唯一渲染身份匹配证明**。不能直接采纳“取center就修好了”，必须在样本中验证世界角点、局部中心、变换逆、实例区域与关联歧义。不闭合对象排除并明确损失支持数。

## 时间与身份

新版仓库 SensorInterface `get_data(frame)` 丢弃非目标frame消息，这是源码级同步意图。可是 `save()` 文件名来自 `self.count`，输出anno字段未保存原World frame、逐摄timestamp或原sensor frame。**同名00000文件不证明原始时间完全一致。** 本轮未取得样本，源时间字段逐项记 `NotAvailable_InReviewedSaveSchema`，不能由10Hz或文件序号编造时间戳后送入HS严格适配器。未来若包内另有原始时间记录，必须以实际证据修正；只有离线序号时，另列弱时序数据合同，禁止当动态安全时序基准。

Actor ID只能在同一次route/session生命周期内使用；静态环境/Actor/Instance编码不必同一namespace。新增0.0.4消失车辆记录有文档声明，但没有载荷审查，不承诺跨帧追踪已通过。不可评价对象、未关联候选与Unknown单独统计，不能算安全Free。

## 依赖与许可

读取一个RGB+json.gz+png不需要安装CARLA、地图、ScenarioRunner或GPU模型。重运行官方采集/220路线则要求CARLA0.9.15、附加地图和独立Benchmark栈；Town11–15资源也不能从当前Town02运行成功推出存在。本轮不部署。

固定 GitHub LICENSE 是 **CC-BY-NC-ND4.0**，HF数据卡写 **Apache-2.0**；这是实测文档差异，不擅自选宽松的一项，也不直接判断所有数据非法。资产级授权范围、衍生数据/发布权限须明确，证据不足时暂停数据使用。本机HF metadata/tree均网络失败（不是认证拒绝证据），web工具可读官方卡；未测试实际数据文件传输。`DatasetSchemaReviewed / AccessAndLicensePartial / NotReadyForIngestion`。
