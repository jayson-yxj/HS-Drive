# J13-A 相机同步、官方候选与公开交付

**结果：候选修复有支持，可准备后续兼容性检查。J13-A 保持 Partial，不授予 SensorSynchronizationPassed。** 官方 0.9.16 候选的 500/500 组元数据一致；有限场景标记检查支持前摄像素与同帧场景状态对应。500 组中 12 组超过原 0.3 秒墙钟门槛并被拒绝，因此不声称持续实时输入通过。模型前向、模型控制、训练、S0/S1/S2 均为 0；移动故障注入和实际车辆停车继续 NotRun。

本报告集中补充此前 0.9.15 诊断，不改写任何历史失败。原始分支及完整私有证据已归档并恢复验证；这里从 origin/main 导出代码和脱敏汇总，没有合并原私有历史。

## 旧异常的准确含义

比较键为 **(run, episode, sensor_generation, camera logical name, sensor actor id, image.frame, image.timestamp, image.transform, width, height, BGRA8)**。完整键值、两条原始记录及生命周期时间线保存在私有 `legacy_conflicts_full.json`，公开汇总见 [旧证据核对](m8j13_sources/upstream/legacy_evidence_audit.json)。

- D2 与 D3 各有六路相机重复；每个重复确实属于同一 sensor actor、同一 generation，frame、timestamp、transform 相同，都是 704×396×4 字节原始 BGRA8。两次最终保存的 BGRA 哈希不同。
- 两条回调都在最后一个排空 Tick 返回后到达，早于 generation 失效、停止监听、Actor 销毁和恢复异步。它们不在正式测量窗口，却复用了已经由 HS 分组器释放的排空组标识。旧流程保留首次来源，将后来记录记为 late，没有覆盖首条。不能把它描述成恢复异步或销毁造成的已证实重复。
- 旧回调只按值复制元数据，**没有独立复制像素字节**。Image 对象保留到清理结束才读取 raw_data 并复制、哈希；缺少入口载荷副本/哈希，不能证明入口至保存期间像素未变。因此旧“相同包头、不同像素哈希”只能指元数据记录与最终载荷副本，不能进一步推断具体拍摄时刻。
- D3 的两次超龄分别为 **0.3771777116 秒、1.2190466654 秒**；起点为同帧 world.tick 调用前 monotonic，终点为六摄最后一条原生回调入口 monotonic。两者同属进程单调墙钟，未与仿真时间相减。旧诊断收齐后释放这些组，仅在事后分析标记超龄；**当时没有实际运行超龄拒绝或车辆保护**。D2 同样有两组 0.3328513969/0.3165468350 秒。旧报告“正常路径应拒绝”不能追认为“该次已拒绝”。

历史同 frame 不同 timestamp 的证据早于 HS 组包：在同一个 Image 直接读出的字段对里已经存在冲突。此前十次启动、三次续办、56 项回归和全部原始失败均保持；原轮次三次启动/892 Tick 单独记账。

## 官方修复与本地构建核对

[CARLA 0.9.16 官方发布记录](https://github.com/carla-simulator/carla/releases/tag/0.9.16)列出了相机 SensorData 与实际图像不匹配的修复。对应提交为 [a095219c77a7710f2f7cbff02043dcb806109b1a](https://github.com/carla-simulator/carla/commit/a095219c77a7710f2f7cbff02043dcb806109b1a)，提交日期 2025-06-18，修改 PixelReader.h、AsyncDataStream.h 和 CHANGELOG.md。只核对 0.9.15→0.9.16 UE4 路径，没有使用 ue58-dev 修复。

旧路径在 render 执行时才读取全局 frame，GPU readback 完成后创建流头，又读取当前 episode 时间与 transform；frame 随后单独覆盖，能造成各字段跨异步阶段取值。修复把 frame、timestamp、transform 一起在游戏线程捕获到 render lambda，再传入 readback payload，使用同一份捕获值构建最终包头。这是游戏线程 capture context 的传播修复，不是客户端用 Snapshot 重写观测值。

三层证据分开：

1. **官方记录了相关修复：已证实**，具体提交和 diff 已保存。
2. **本地具有同类缺陷：原生字段冲突已实测，与旧 tag 路径相容。** 本地 0.9.15 二进制与保留发布包内成员哈希一致，没有相机二进制被本地替换的迹象；但没有对发布二进制加 trace，不能声称已定位运行时第一处错误指令。
3. **候选有效性：仅对本轮完整 0.9.16 包和官方同版本客户端有有限支持。** 版本之间还有其他变化，不能当作单独补丁的因果验证。

本机已有官方 Linux 包 8,346,095,504 字节，SHA256 `09e3ebb28df17962f0c997e66f4b914ad5ea6f1d6a6dbbf13c9f87eb38346d57`，安装约 19 GiB，核对时可用空间约 117 GiB。重算整个包哈希并比较包内二进制与安装二进制一致；本轮下载字节为 0，无附加地图、UE 编译、驱动或模型环境变更。

候选使用已有 `.venv-m6f-0916` 官方 PyPI 0.9.16 客户端；其 libcarla 与该安装的 wheel RECORD 一致。发布包附带的同版本 wheel 与 PyPI wheel 二进制不同，**不宣称两者逐字节相同**；本轮始终使用同一个已记录哈希的 PyPI 客户端。客户端/服务器均实际报告 0.9.16。详见 [资产及上游审计](m8j13_sources/upstream/upstream_audit.json)。原 0.9.15 路径、客户端、配置和复现文件未改。

## 两次候选运行

配置冻结在 `configs/m8j13_candidate.json`。沿用 Town03、自车车型、六摄名称/顺序/外参/704×396/FOV、sensor_tick=0、Epic 离屏渲染、0.05 秒物理步长。单一 Tick 所有者，20Hz 墙钟目标且不突发补拍；按原始 frame 关联 Snapshot，不使用旧轮次的逐组等待屏障。该调度差异与版本差异均披露，不把结果改善归因给单一因素。

|运行|固定窗口|MetadataConsistency|CaptureAssociation|DeadlineAndProtection|
|---|---|---|---|---|
|R1|20 初始化 + 80 测量 + 6 排空|80/80 完整；无元数据冲突|80/80 前摄标记匹配；7 次实测左右切换均匹配，其中 6 次在测量窗口|6/80 超龄拒绝，生成 6 个未施加停车输出|
|R2|40 初始化 + 500 测量 + 6 排空|500/500 完整；无元数据冲突|与 R1 有限证据一致；本轮没有标记，也没有新增重复|12/500 超龄拒绝，488 接受，12 个未施加停车输出|

初始化和排空也核对：R1 全部 106 组六摄、R2 全部 546 组六摄均有对应 Snapshot，frame/time 最大误差不超过原容差，位姿矩阵最大差为 0；两次全阶段没有重复键、缺失组或队列溢出。全部原始记录按追加列表保存，不覆盖重复键。完整分组的分母始终为预先固定窗口。

两次共 **652 次显式 Tick + 60 次 ApplySettings 内部 Tick 保守计费 = 712/900**。没有同步 reload；地图只在异步初始化时按需加载一次。旧三次/892 Tick 不重置、不抵扣。运行后同步设置恢复、传感器/Actor 销毁、进程组退出、2300–2302 端口释放。额度用完，不追加第三次。

### CaptureAssociation 的范围

R1 使用真实场景中的红色 Tesla 作为诊断标记，固定在前摄视野内两侧位置，按预定不规则左右序列每 10 Tick 切换，自车保持静止。标记从未写入收图后的像素；每帧使用 WorldSnapshot 中实测 marker pose 预测图像 ROI，以冻结的红色阈值核对目标侧与对侧。命令请求位置不替代 Snapshot 实测位置。

80 个测量帧全部匹配；切换帧的目标侧红色像素明显高于对侧，上一侧假设不满足固定判据。人工检查一张私有原图确认是场景内标记。新回调入口将 raw_data 独立复制为不可变 bytes，然后才入队；主线程从该副本哈希、分析和私存，不依赖销毁后的 Image 对象。

这些结果支持**前摄有限标记状态在切换时与同帧 Snapshot 对应**。它不是所有像素、全部相机、所有运动状态的绝对拍摄时刻证明；R2 未新增标记，不把“没有重复”单独作为像素关联通过依据。官方 capture context 修复与该有限实验共同支持候选，而非无限证明。

### DeadlineAndProtection 的范围

保留仿真/墙钟各 0.3 秒的原合同。仿真年龄为当前 Snapshot elapsed − 输入 Snapshot elapsed；墙钟年龄为最终分组/检查完成的 monotonic − 同帧 Tick 发起前 monotonic。新判定终点比旧回调入口统计更晚，含按值复制、队列、哈希和检查；因此不能把两轮延迟数字直接当作模型速度对比。

R2 最终判定墙钟年龄 P50/P95/P99 为 **157.7/188.9/477.8 ms**，最大 **1310.6 ms**。超时是实测结果，门槛不调整。`freshness` 返回 PlanWallExpired 后，该组标记 rejected，并调用独立 SafetyGate.stop 产生 `{steer:0, throttle:0, brake:1}`，不发给车辆；测量输入无需控制，继续收集剩余固定窗口以保留分母。**数值停车输出不等于实际车辆停车通过**。两次全阶段超龄分别为 12/15 组，测量窗口为 6/12 组。

详见 [R1](m8j13_sources/upstream/R1_summary.json)、[R2](m8j13_sources/upstream/R2_summary.json)、[验收汇总](m8j13_sources/upstream/acceptance.json)。

## 公开代码与私有历史

用户指定公开分支 `feat/sparsedrivev2-carla-closed-loop-public`。该分支从干净 origin/main 创建，按允许列表导出代码、测试、配置和重写的脱敏汇总；未 merge/cherry-pick 原私有分支。原始完整 SHA、导出提交映射、完整 bundle、原始报告和日志保存在私有交接目录。bundle 克隆恢复后验证原分支 tip，证据归档解包后核对 252 个文件；原历史没有改写或销毁。

保留历史 56 项回归结论；候选新增入口字节复制、独立时钟超龄拒绝和标记辨识测试。最终回归结果及新增历史审查见 `reports/m8j13_sources/upstream/regression.json` 和 `public_history_review.json`。公开推送只包含指定分支的新增可达历史，不推原私有分支或全部 refs。main、原配 PID、模型配置、参考权重保持。

后续可以准备 0.9.16 与 Bench2Drive/HS 的兼容性检查；没有自动升级正式基线，没有放行模型驾驶。仍需处理截止期尾延迟、移动故障注入和实际停车保护等 J13-A 未完成项。
