# M8-I1b 根因假设与退出判断

**唯一决策：B — `StartupRootCauseUnresolved`。** 现有证据能够排除若干错误解释、区分启动阶段，但不能将失败收敛到一个有明确依据、值得再消耗一次启动额度的原因。本轮不提出新的启动实验，不靠扩大超时或改参数碰碰运气。

## 不能合并的两次失败

M8-I1 的实际异常为 `MapMismatch_NoReload`，执行路径已通过0.9.15版本检查并读取过World/Map；实际地图名未落盘，XODR检查在地图拒绝之后，未执行。不能把默认Town10HD_Opt填成实际地图。

M8-I1a 有Shipping进程、TCP监听证据，但版本RPC连续31次超时。未调用World/Map身份读取，地图名、Episode、Frame、XODR均未知。`TCPListening`不能升级为`CARLAVersionRPCReady`，也不能把此故障称作已证实的地图加载失败。

## 假设审查

| 假设 | 支持的事实 | 缺失或反证边界 | 决策 |
|---|---|---|---|
| Town02命令行参数未生效 | I1地图检查拒绝；argv确有Town02；默认配置另指Town10HD_Opt | I1实际名缺失；I1a未达地图读取；没有本轮解析/加载日志 | 未确定，不能统一解释两次失败 |
| CWD导致资源定位错误 | H1脚本cd安装根；I1未保存实际CWD；I1a包装脚本在仓库、binary在Binaries/Linux | H1/I1二进制CWD未记录，没有资源路径失败日志；UE进程可自行改变CWD | 未确定，不能以当前cwd补写历史 |
| RPC超时2s过短 | H1 preflight设60s；I1监督器2s成功后Runner设30s；I1a2s失败31次 | I1同样2s能够版本就绪；没有I1a排队/服务时间测量；长窗口反复失败不是单次时延证明 | 不支持盲目增加timeout |
| 重用Client导致超时后连接状态异常 | I1监督器每次循环新建Client；I1a复用一个Client | 没有客户端内部状态、服务端请求轨迹或配对实验，差异本身不是错误来源 | 不支持凭这一差异申请启动 |
| H1已证明Town02原生直接启动正常 | H1最终Map=Town02且API XODR匹配 | H1启动argv无地图，preflight允许条件load_world；该分支实际是否执行未记录 | 该比较结论无效；H1仅证明当时最终世界可用 |
| GPU/内存/权限故障 | I1a单点GPU99%、6004MiB；stdout有EOF/stream退出文字 | 相同窗口无可关联Xid/OOM/权限日志；EOF无逐行时间且可能来自TCP探针/清理；99%不是错误 | 不支持EnvironmentIssueSupported |
| 旧CrashContext证明本轮崩溃 | 找到PID799952、signal11记录 | 日期10月19日、port2000、Low，与目标日期/PID/端口/质量不符 | 明确排除，不可挪用 |
| CrashReportClient.ini证明相应轮次崩溃 | 每轮启动附近存在该配置文件mtime | 内容只有通用配置，没有错误、PID、Map或CrashContext | 只支持时间相关的配置写入，不证明崩溃 |

## 日志与系统证据限制

安装包Saved/Logs及相关用户路径没有可归属的独立UE启动日志，状态`NoRunMatchedUELog`；两轮自有stdout仍以运行目录、Popen源码和SHA关联，不能说所有日志都不存在。H1没有可核验的原始server stdout，最终成功World结果可验证，但其最初启动地图和就绪总耗时不能重建。

时间窗口由原始文件mtime、Supervisor时长、已有进程stat和OS探针组成。mtime只是窗口辅助，不是内部事件时间；Host dmesg时钟换算和容器PID命名空间亦有限制。I1/I1a窗口内未发现相关内核错误，journal存储缺失；这不是系统绝无错误的证明。

## 退出安排

停止CARLA启动器修补循环，下一项建议为 **Bench2Drive单片段许可与数据访问条件评审**，复用M8-I0已记录的问题清单，先确认资产许可、官方入口与能否合法取得一个有限片段。本轮不重新下载或访问数据、不进行任何采集或220路线执行，也不把这一建议当作新任务批准。

只有未来出现新的可核查日志或独立证据才重开启动根因判断；本轮不提出“再试一次”的单变量方案，不把失败传播和清理通过转化为启动、训练或安全门禁通过。
