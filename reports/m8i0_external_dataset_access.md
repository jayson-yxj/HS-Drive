# 外部数据取得与 nuScenes 参考

检查日期2026-10-26。所有本机HTTP失败类别、固定URL、响应SHA见 `m8i0_sources/*.meta.json`；网页工具摘录另存 `web_dataset_cards.json`，与本机文件下载证据分开。

GitHub API允许取得固定commit源码；raw.githubusercontent.com数次Timeout/Network unreachable，改用同一官方repo/commit Contents API取得，base64解码后验证Git blob SHA。没有更换第三方镜像，没有关闭TLS，没有输出代理凭据。错误记录保留，404为路径不存在；例如ScenarioRunner行人配置实际在 `ObjectCrossing.xml`，不是猜测的 `DynamicObjectCrossing.xml`。公开源码下载不等于框架安装。

Bench2Drive本机HF六次metadata/tree请求均失败，网页工具可读官方卡。故不能声称GPU服务器能下载数据，不能把网络错误解释为权限/配额问题。旧Mini官方清单有逐包SHA，可为以后单包传输校验；本轮载荷下载0。主数据约400G、新Depth约7.3T均不属于自动授权范围。GitHub/HF许可差异详见 [合同](m8i0_bench2drive_data_contract.md)。

条件备用只提出一个旧Mini片段：`DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz`，280,176,944 bytes，官方清单SHA `b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2`。理由是行人/遮挡源码支持，不是按PETR预测挑选。先澄清许可及官方端点可达，再单独批准最多一个包、先一帧Schema/标定检查；不运行 `download_mini.sh` 一次取全部10段。Town02与HS历史相关，始终Development，不叫新Final。包若缺原时间或身份无法解释，停止，不自动换包。

## nuScenes：只作原训练域参考

固定devkit `b40adc467b919192899405d9b77871afee8efa07` 的 [Schema](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/docs/schema_nuscenes.md) 提供：calibrated_sensor（sensor到ego的平移/quaternion与intrinsic）、sample_data（独立timestamp及ego_pose引用）、全局sample_annotation（center、w/l/h、quaternion、instance_token、visibility）。每个传感器以各自原时间/ego pose组合，不能伪造CARLA完全同帧；global→ego→camera与HS左右手轴必须独立核验。

官方 [教程](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html) 展示Mini及六摄记录，[论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf) 说明相机1600×900、12Hz与2Hz关键帧；真实采集不是六路CARLA同步Tick。Mini可用于原PETR配置、Box底部/中心和参考坐标诊断；不能替代CARLA域质量，更不能与HS历史当同分布。

官网当前主页/条款web读取只返回JavaScript壳，完整条款与账户访问资格本轮 **NotAvailable/NotVerified**；devkit README不是数据授权。遵循 [官方网站](https://www.nuscenes.org/) 的注册/条款/授权流程，不绕过。没有下载Mini，没有运行PETR。若未来合法取得，原域冻结推理必须另行申请；本轮不将nuScenes列第二备用路线。
