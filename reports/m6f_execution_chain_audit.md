# M6-F 执行链审计

研究范围为Privileged仿真观测，视觉不控制车辆。主系统CARLA0.9.15、Tesla P0、Oracle Planner/Safety及六秒预测合同未改。

## 数据流及可观测边界

`cycle_command请求 → Vehicle.apply_control RPC → LastAppliedControl → UDefaultMovementComponent.ProcessControl → UE车辆移动组件 → PhysX → ActorSnapshot`

0.9.16 `get_telemetry_data`在CARLA Server通过`BIND_SYNC`调用`FCarlaActor.GetVehicleTelemetryData`，最终读取`ACarlaWheeledVehicle.GetVehicleTelemetryData`。车身速度、Gear和RPM来自移动组件；轮速来自`mWheelsDynData`，滑移及摩擦来自受读锁保护的`PxWheelQueryResult`；载荷、力、扭矩来自`UVehicleWheel.Debug*`字段。[官方车辆实现](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp)

原始Throttle/Brake会传到UE的`SetThrottleInput/SetBrakeInput`，Gear由自动/手动配置及`SetTargetGear`管理。CARLA Applied Control并非PhysX最终平滑后的模拟输入，命令一致不能证明内部执行完全一致。[官方转换路径](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/MovementComponents/DefaultMovementComponent.cpp)

`torque`只可命名为真实API返回的**调试轮端扭矩**，不是实际刹车扭矩。没有扭矩分解、接触冲量、离合器状态和求解器子步。不能用`4*T*brake/(m*r)`替代内部测量，也不把`sum(long_force)/mass`等同全部车身加速度。

## 时间与采样

ActorSnapshot具有frame和elapsed_seconds，Telemetry结构没有这两个字段。每个RPC前后获取Snapshot，独占同步Tick且帧号相同才关联；下一tick必须连续。第一轮还在没有tick的情况下重复请求，检查字段是否变化。这里证明的是可重复的外层帧边界读取，不证明Debug字段来自同一个PhysX子步。最后子步瞬时/调试力和跨50ms速度差分可不一致，不能据此直接判定引擎违反动力学。

原始车身速度、API加速度和未经滤波差分加速度/Jerk同时保留。Jerk是50ms采样指标，不是连续求解器峰值。零载荷除法产生的非有限归一化力用`invalid_raw`保留，不伪造成0。

## PhysicsControl干预

官方源码在ApplyVehiclePhysicsControl中明确调用`RecreatePhysicsState()`，再设置轮端参数、重设Constraints。因此“参数相同”不保证传动/轮端隐状态相同。[官方Physics应用路径](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp#L600)

本轮若运行Sham，固定首次spawn tick之后、初始化之前的施加时机，比较未施加、原参数重新施加、历史半制动扭矩三组；第二重复反转顺序。没有修改移动中Physics或搜索新参数。即使本小样本未发现差异，也不能证明重建状态是恒等操作。M6-E P1紧急距离退化和NoGo保持。

## 源码与版本可信度

本地Python项目模块由`m6f_launch.CapturedSource`直接读取字节并编译，保存导入时source SHA和实际module code object SHA；批次末核验磁盘未变。标准库/第三方模块另列磁盘边界哈希，不冒充进程内存完整哈希。Launcher自身保存执行code object哈希。每批独立进程、提交、启动清单，失败不覆盖。

服务端归档SHA、二进制SHA、VERSION、Python安装清单单独冻结。官方Git tag用于解释源码，发行二进制与全部Unreal源码的精确构建对应关系未获完整可重复构建证明。0.9.16与0.9.15公开Physics相同不能消除引擎/包/地图版本影响。

## 0915备选插桩

本轮已获得0.9.16真实轮速/滑移及调试力路径，不同时开发0915原生插桩。0915现有SDK不提供本RPC。完整制动扭矩/子步接触因果分析仍需要更细插桩，但没有合法可用且与运行包匹配的Unreal源构建环境证据；不声称已完成，也不启动全服务器重编译。
