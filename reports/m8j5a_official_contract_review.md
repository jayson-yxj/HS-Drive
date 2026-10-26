# 固定官方标注合同与实际载荷核对

日期：2026-10-26 UTC。唯一官方代码版本 `7ec25d1c9f7522d923ce5f3420986cef1cb2d956`。复用此前冻结源码，并有界取得该版本的可视化脚本；未下载新数据。来源 SHA 见 `m8j5a_conversion_validation.json`。

官方入口：[anno.md](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/anno.md)、[data_collect.py](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/data_collect.py)、[visualize.py](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/visualize.py)、[utils.py](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/utils.py)。

| 项目 | 官方及载荷证据 | 审计处理 |
|---|---|---|
| extent | 半长、半宽、半高 | 尺寸用两倍 extent；与八角点轴长独立比对 |
| center | collector 将局部 bounding_box.location 变换到世界 | 不用 Actor origin 替代，也不再把世界 center 乘一次 Actor-to-world |
| bbx_loc | 文档坐标说明不够准确；collector 保存 Actor 本地 bounding_box.location | 以实际构造式及已记录 world_cord 互证，不按字段描述猜变换 |
| 动态车辆 | `*vehicle*` 范围 100m；保存完整世界角点、逆矩阵、state | 保留范围内原始记录；dynamic 不等于每帧都运动 |
| 静态车辆 | `*static.prop.mesh*` 特定名称；与 level BB 近邻匹配，存在跳过条件；官方警告 location/rotation 异常 | 优先 world_cord；缺失时用可靠 bbx_loc、extent、逆 world2vehicle；不使用异常 location/rotation 重建。缺少局部参考则 Unknown |
| 行人 | 活跃 `walker*`、100m 范围，异常分支可跳过 | 原始不存在不自动视作背景；源地下位置不强行移动到地面 |
| 灯/标志 | level BB、trigger 信息，部分标志是地面区域 | 显示代理与训练目标分开；不把 trigger 当作物理障碍盒 |
| 相机 | 从 RGB sensor 实际 pose 保存 world2cam；安装参数保存 cam2ego；各自 K | 两条独立投影路径对照，保持源标定；未套用 HS 自车安装参数 |
| 光学坐标 | CARLA camera 的 (x,y,z) 转为光学 (y,-z,x) | HS Ego 为 (x,-y,z)，保持完整矩阵链 |
| Semantic/Instance | 原始传感器 BGRA 信息；相同相机安装 | Instance R 是语义，G/B 组合仅作 render token；与独立 Semantic 及几何交叉验证，不等于 Actor ID |

静态说明存在歧义：文档同时建议用 center/extent 得到局部 Box 后再逆变换，而 collector 实际保存世界 center 和 Actor-local bbx_loc。此次实现用源码构造式和角点一致性消歧；不对世界 center 二次变换。片段没有任何 `state=static` 的原始条目，静态路径只有合成回归证据。

官方可视化还有额外展示裁剪：75m、相对高度、中心视线角度等；标志显示会放大一轴 extent。它适合参考投影方法，不适合证明标签完整。本次全原始层不沿用这些展示裁剪，也不修改原始 extent。原始八角点保留完整 3D 姿态；裁剪仅针对相机近面和图像边界，不是遮挡认证。

官方文档中的行人速度为零提示也不能机械套用：本片段实际存在非零 walker speed。固定官方代码版本说明不是该压缩包逐字节生成代码证明。原始逐传感器时间戳缺失，不能推断同序号所有输出绝对同步；`00124` 缺口的生命周期/同步解释尚待证据。

未使用 LiDAR 点数、Radar、Depth uint8、Top-down 或 expert_assessment 补造标签。未重新解释旧深度为高精度监督。所有新结果只用于离线审计，不进入 RGB 推理输入。
