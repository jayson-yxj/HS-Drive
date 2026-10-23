# M6-G 官方源码与运行证据调用链

**离线源码审计。没有编译、没有获取新的内部遥测。** CARLA官方0.9.16解析到`294096eb1c38eabf246e4f3a9cdab704e33a7f4c`。NVIDIA公开PhysX3.4机制参考固定到`5e42a5f112351a223c19c17bb331e6c55037b8eb`。本地完整来源、文件SHA和Git blob见`m6g_sources/manifest.json`及supplement/flush补充清单。

**二者均不是安装Shipping完整可重复构建身份证明。** Unreal4.26 CARLA fork源码无法访问，尤其Tesla四轮UE移动组件、轮胎回调到Debug字段的写入时序仍有缺口。公开CARLA中的NW实现只作“该架构存在平滑路径”的例子，不当作Tesla4W已执行证据。不以PhysX5文档替代旧版实现，不使用未授权UE镜像绕过访问。

## 请求保存、应用与回读

| 层级 | 精确文件/类/函数 | 可证明的源码行为与限制 |
|---|---|---|
| 客户端请求 | `LibCarla/source/carla/client/Vehicle.cpp`，`Vehicle::ApplyControl` | sticky_control可避免重复发送相同命令；客户端保存上次control用于比较；不是服务端应用确认 |
| RPC | `LibCarla/source/carla/client/detail/Client.cpp:515`，`Client::ApplyControlToVehicle` | 使用`AsyncCall`；返回不证明命令已作用于当前物理帧。`GetVehicleTelemetryData:505`使用`CallAndWait` |
| 服务端路由 | `.../Server/CarlaServer.cpp:2145`，`BIND_SYNC(apply_control_to_vehicle)`；`.../Actor/CarlaActor.cpp:865`，`FVehicleActor::ApplyControlToVehicle` | 查Actor并转发Control/Priority，休眠Actor有另外保存路径 |
| 请求缓存 | `.../Vehicle/CarlaWheeledVehicle.h:211`，`ACarlaWheeledVehicle::ApplyVehicleControl` | 满足Priority时写`InputControl.Control`，原始control关闭Ackermann模式 |
| Tick消费 | `.../Vehicle/WheeledVehicleAIController.cpp:131–146`，`AWheeledVehicleAIController::Tick` | 有效Vehicle在该Tick调用`FlushVehicleControl`；源文件见`flush_corrected_manifest.json`，初次误用文件名的404保留 |
| 外层应用 | `.../Vehicle/CarlaWheeledVehicle.cpp:334`，`FlushVehicleControl` | 调`BaseMovementComponent->ProcessControl`，回写reverse，之后复制到`LastAppliedControl`。该记录不等于最终PhysX模拟输入 |
| UE入口 | `.../Vehicle/MovementComponents/DefaultMovementComponent.cpp:24`，`UDefaultMovementComponent::ProcessControl` | SetThrottleInput/SteeringInput/BrakeInput/HandbrakeInput；按reverse/manual选择自动挡和SetTargetGear，最后读取CurrentGear |
| 客户端回读 | `Vehicle::GetControl` / `CarlaWheeledVehicle::GetVehicleControl` | 客户端从ActorSnapshot取control；服务端getter返回LastAppliedControl。Telemetry也读LastAppliedControl，因此一致不是独立的执行器输入测量 |
| 仿真调度 | `.../Game/CarlaEngine.cpp:299`，同步RPC循环及`UpdateFrameCounter`；`:397`子步配置 | tick cue驱动外层推进，配置UE MaxSubstepDeltaTime/MaxSubsteps；没有给本轮遥测新增求解器时间戳 |

[CARLA客户端](https://github.com/carla-simulator/carla/blob/294096eb1c38eabf246e4f3a9cdab704e33a7f4c/LibCarla/source/carla/client/Vehicle.cpp)、[RPC实现](https://github.com/carla-simulator/carla/blob/294096eb1c38eabf246e4f3a9cdab704e33a7f4c/LibCarla/source/carla/client/detail/Client.cpp)、[Control缓存定义](https://github.com/carla-simulator/carla/blob/294096eb1c38eabf246e4f3a9cdab704e33a7f4c/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.h)。

本轮重现25个晚1帧匹配事件，全部以首次观察到Applied的帧对齐。源码支持“请求返回不是应用ack”这个机制边界，不能仅据AsyncCall认定每一个延迟的线程调度根因。实际作用时刻只在相邻外层快照之间有界，**第一Applied观测时间也不是精确子步作用时间**。

## 输入平滑、传动与最终求解输入

CARLA `WheeledVehicleMovementComponentNW.cpp:273` 的`UpdateSimulation`将`ThrottleInput/BrakeInput`装入RawInputData，在`:304–310`传Rise/Fall Rate并调用`PxVehicleDriveNWSmoothAnalogRawInputsAndSetAnalogInputs`。NVIDIA `VehicleUtilControl.cpp` 的`processAnalogValue:128`和`PxVehicleDriveSmoothAnalogRawInputsAndSetAnalogInputs:182`提供按dt及变化率更新模拟输入的机制。**Tesla实际4W对应UE实现、参数以及是否还有额外Brake重映射未取得，NotAvailable**；不能把NW路径或默认值套成Tesla实测。

NVIDIA `PxVehicleUpdate.cpp` 的4W update读取`mControlAnalogVals`，在`:4980`附近执行`processAutoBox`和`processGears`。`processAutoBox:1270`检查current/target gear、自动换挡计时及normalized engine speed，上/下挡阈值；换挡中可抑制加速输入。`processGears`描述经过neutral、等待switch time后到target gear的流程。**这些内部target/timer没有出现在M6F遥测中**；冷启动3.55s的外层等待不能唯一分解成哪一项内部计时。不要把CARLA Gear0直接套成PhysX枚举的整数下标，二者编码不同。

[CARLA NW实现](https://github.com/carla-simulator/carla/blob/294096eb1c38eabf246e4f3a9cdab704e33a7f4c/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/WheeledVehicleMovementComponentNW.cpp)、[NVIDIA输入平滑参考](https://github.com/NVIDIAGameWorks/PhysX-3.4/blob/5e42a5f112351a223c19c17bb331e6c55037b8eb/PhysX_3.4/Source/PhysXVehicle/src/VehicleUtilControl.cpp)。

## 刹车扭矩、车轮锁止与低速约束

以下均为**PhysX3.4公开源码机制支持，安装二进制因果未验证**：

- `computeBrakeAndHandBrakeTorques:1894`用最终brake/handbrake、轮速符号及wheel max torque生成每轮局部`brakeTorques`，另有`isBrakeApplied`。读取这个求解器局部量及输入来源，才可能得到有验证来源的刹车扭矩；Python `DebugWheelTorque`不是它。
- 4W角速度求解后`:3873–3883`存在“制动且新旧角速度乘积≤0，置轮速0”的限制，源码注明锁止可持续到释放制动。这个分支可以由很小的非零brake启用，不能只用最大扭矩线性比例推断最终车轮响应。是否在本Shipping事件真正命中仍Unresolved。
- `computeTireSlips:1990`根据制动/加速状态及速度分母计算滑移；`processSuspTireWheels`计算轮胎载荷/力。`activateStickyFrictionForwardConstraint:2150`按接触前向低速、锁止及加速意图或低速计时决定低速约束；`:3459`激活后将longSlip置0，由约束承担该方向速度处理。
- `PxVehicleComputeTireForceDefault:2230`输出轮胎力和轮端反作用矩，`:2288`的关系为负纵向力乘轮半径。它解释M6F debug torque/force数值关系的候选来源，**不是Brake Torque实测**。

[上述机制的固定NVIDIA源码](https://github.com/NVIDIAGameWorks/PhysX-3.4/blob/5e42a5f112351a223c19c17bb331e6c55037b8eb/PhysX_3.4/Source/PhysXVehicle/src/PxVehicleUpdate.cpp)。这说明“末帧零调试力而车身仍变速”有合理的机制候选，不证明全部17个小制动峰值均由sticky约束造成。之前子步力、接触冲量、输入平滑与其他约束也可能有贡献。

## 子步、QueryResult和Debug更新时间

公开PhysX4W update在`:4630`按前向速度选择车辆内部substep数；`:5096–5106`使用`subTimestep=timestep/numSubSteps`及循环k。这个**车辆内部积分子步**与UE scene physics substep、CARLA外层frame是三种不同时间层级，不能直接把50ms除以配置上限当真实子步序列。

每个车辆内部子步先计算brakeTorques、`processSuspTireWheels`，再`storeSuspWheelTireResults`，之后角速度积分更新。部分约束数据取第一个子步，部分缓存只取最后一个子步。这意味着一次API结构中的QueryResult和更新后wheel omega也未必表示完全同一计算相位。

CARLA `GetVehicleTelemetryData:829`从受scene读锁保护的`PxWheelQueryResult`读slip/friction，从`mWheelsDynData.getWheelRotationSpeed`读omega，从`UVehicleWheel.DebugTireLoad/DebugLongForce/DebugWheelTorque`读debug值。力除100、扭矩除10000转换为N/Nm；omega为rad/s，latSlip转degrees。**UE实际写Debug字段的回调及刷新相位未能审查**，不能把公开PhysX graph debug记录函数误认成UE `DebugLongForce`写入点。API结构没有frame、substep或timestamp，外层RPC帧bracket不能填补此缺口。

[官方Telemetry实现](https://github.com/carla-simulator/carla/blob/294096eb1c38eabf246e4f3a9cdab704e33a7f4c/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp#L829)。

## Physics重建

`ApplyVehiclePhysicsControl:600`写参数、wheel setup，在`:683–686`锁scene并调用`RecreatePhysicsState`，随后再写PhysX wheel/tire参数并`ResetConstraints`。公开调用点证明不是仅修改一个浮点数；UE实现缺失，无法给出完整隐状态重置列表。

已有Sham前后live速度复位，本轮另发现首次tick垂直速度比未重建组大0.49m/s、z高0.0245m，两个重复一致；说明坠落/初始化过程受重建影响。它是实际车身观测，不能冒称直接读到内部线速度重置指令或证明所有后续制动差异的唯一来源。

## 插桩准入及停止

官方构建要求可访问CARLA修改版UE4.26 fork。当前无本地UE源码，官方访问未成功，未配置UE4_ROOT，缺少匹配UE工具链；虽然磁盘超过官方130GB参考，仍无法构建可比基线。**Instrumentation_NotRun_BuildPrerequisiteFailed**。不下载非官方UE镜像，不运行完整编译，不修改Shipping二进制。

仅当将来独立具备合法匹配源码与构建身份时，才可能在最终模拟输入、`brakeTorques`局部量、gear target/timers、sticky active及车辆内部k/dt处做观测。必须同时标明UE scene step与车辆内部step，配对官方/观测版验证行为和开销。这个是有证据缺口指向的候选插桩点清单，**没有实现、没有批准自动继续**。
