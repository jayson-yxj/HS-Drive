"""Report actual early-stop results without manufacturing downstream performance."""
import json
from pathlib import Path
import numpy as np
from tools.m8h1_common import *
def md(p,s):
 with Path(p).open('x') as f:f.write(s)
def main():
 result=json.loads((ROOT/'actor_smoke_results.json').read_text());ego=[]
 for t in result['templates']:
  r=json.loads(Path(t['reviews']['ego']['path']).read_text());ego.append(dict(template=t['template'],source=t['reviews']['ego'],spawn_status='SpawnSucceeded',settling_status=r['settling']['status'],geometry_status=r['geometry']['status'],stationarity=r['settling']['stationarity'],drift=r['settling']['drift'],final_pose=r['settling']['final_pose'],final_OBB=r['settling']['final_OBB'],stop_reasons=r['geometry']['reasons'],camera_pose_closure='NotRun_NoSensors',original_static_request_result=r['geometry']['original_static_request_result']))
 assert len(ego)==2 and all(e['settling_status']=='SettledPoseObserved' and e['geometry_status']=='GeometryNoGo' for e in ego)
 write('reports/m8h1_ego_settling.json',dict(**provenance('ActualCARLANaturalPhysicsSettling'),rows=ego,valid_Ego_spawned=2,stationary_observed=2,geometry_proxy_supported=0,invalid_startup_attempt=asset(ROOT/'invalid_measurement_attempt/reason.json'),all_spawn_calls_including_invalid=4,initial_rpc_pose_vs_snapshot='Pre-tick getter was zero; invalid attempt preserved. Corrected run uses frozen spawn RPC for drift reference and first tick snapshot for observed state.',physics_never_disabled=True,not_exact_contact_certification=True))
 for filename,role in [('vehicle_settling','target vehicle'),('walker_grounding','target walker')]:
  write('reports/m8h1_'+filename+'.json',dict(**provenance('NotRun_PreconditionFailed'),status='NotRun',role=role,objects_spawned=0,reason='Both Ego failed unchanged15point support evidence gate; stop before generating downstream targets',templates=[dict(template=e['template'],status='NotRun',reason='EgoGeometryNoGo') for e in ego],no_claim_of_success=True))
 for filename in ['actual_visibility','render_identity','counterfactual_identity','sensor_synchronization']:
  write('reports/m8h1_'+filename+'.json',dict(**provenance('NotRun_PreconditionFailed'),status='NotRun',reason='No target actors or sensors created after Ego geometry failure',new_keyframes=0,new_RGB=0,sensor_payloads=0,target_objects_evaluable=0,unobservable_target_not_counted_as_detection_failure=True,physical_false_positive_claim=False,templates=[dict(template=e['template'],status='NotRun') for e in ego]))
 write('reports/m8h1_training_readiness.json',dict(**provenance('IndependentTrainingDecision'),status='TrainingReadinessNoGo',split='DevelopmentOnly',history='HistoricalCoverageIncomplete',FinalEligible=False,new_Final=0,natural_target_support=0,new_training_samples=0,training_authorized=False,visual_control_authorized=False,reason='Two-template core actual-geometry precondition failed; no identity or sensor evidence'))
 write('reports/m8h1_acceptance.json',dict(**provenance('ActualBoundedSmokeDecision'),status='DevelopmentSmokeNoGo',Ego_spawn='ActualSpawnGo',Ego_stationary='SettledPoseObserved',Ego_geometry='NoGo_UnresolvedSupportEvidence',target_vehicle='NotRun',target_walker='NotRun',shifted_preflight='ShiftedContractGo',shifted_actual='NotRun',render_identity='NotRun',sensor_sync='NotRun',four_phase='NotRun',training='TrainingReadinessNoGo',historical_coverage='HistoricalCoverageIncomplete',FinalEligible=False,valid_Ego_spawned=2,invalid_startup_spawned=2,total_spawned=4,targets_spawned=0,sensors_spawned=0,keyframes=0,RGB_images=0,payloads=0,completed_templates=0,required_templates=2,original_thresholds_unchanged=True,automatic_placer_expansion='Stopped',next_architecture='Independent review of existing CARLA traffic scenes / ScenarioRunner / mature annotated data; no M8-H2 ray threshold patching',default_control_and_models='Unchanged'))
 md('reports/m8h1_summary.md',f'''# M8-H1 — Native Curated Templates Actual Actor Smoke

结论：**DevelopmentSmokeNoGo / TrainingReadinessNoGo**。真实执行了两个冻结模板的 Ego 生成与自然落稳；均达到静止标准，但均未通过原支撑证据合同。按门禁停止，目标车辆、行人、传感器及四阶段采集全部 **NotRun**。不得将缺少的图像/身份结果写为通过或检测失败。

基础 `{BASE}`，分支 `feat/m8h1-native-template-actor-smoke`。原始证据 `{ROOT}`，报告带执行父提交与逐文件源码SHA，最终commit封存manifest。

## 执行与前置合同

只使用模板00（spawn044/049、nav07）和02（spawn100/045、nav33）。M8-H0原始Transform、Rig、XODR、地图包、导航及服务器资源SHA均一致；实时静态环境库存与H0相同。没有新随机导航调用，没有使用模板01/03。

每个目标只生成一个沿原声明车道 `next(3m)` 的 Shifted候选，保持原横向偏移、道路/车道/section；车辆保留原生+0.5m高度偏移，行人依据15条Sidewalk射线和历史local Box底部推导初始中心。四个Shifted均通过原始预检；Present/Shifted所有对象间OBB关系及历史足迹记录已保存。**预检通过不代表Shifted真实生成/身份闭合通过。**

静止协议在实际生成前冻结：dt=0.05s，最多200Tick、至少40Tick，末20Tick原始速度≤0.02m/s、位置范围≤0.01m、姿态范围≤0.10°；原0.02m横向漂移/0.5°最终姿态约束不变。仅固定制动和手刹，未启动驾驶控制器，未禁用或改动Physics。

## 实际结果

| 指标 | 模板00 | 模板02 |
|---|---:|---:|
| Ego真实Spawn | 成功 | 成功 |
| 达到静止窗口 | 40Tick / 2s | 40Tick / 2s |
| 相对原生Spawn的Z变化 | −0.278387m | −0.278387m |
| 最终Z | 0.221613m | 0.221613m |
| XY漂移 | 0m | 0.0000305m |
| 末窗口最大速度 | 0.0028703m/s | 0.0028700m/s |
| 原支撑射线有效数 | 0/15 | 0/15 |
| Ego完整几何资格 | NoGo | NoGo |
| 目标车辆/行人、RGB | NotRun | NotRun |

停止的主要原因是 **支撑证据不可取得，而不是观测到车辆继续滚动或确认悬空**。原 `actual_check` 从OBB底部向下5mm开始射线。实际OBB底部约0.21364m，原地面观测约0.22m；射线起点约0.20864m，已经低于地面约11.36mm，因此向下射线没有命中。底部与先前地面差约−6.36mm本身在原−30mm容差内，但先前地面不能冒充当前实际支撑命中。15个点均Unknown，门禁不能关闭。没有上移射线或修改通过语义后再跑。

另有实际local Box与历史世界坐标反算local Box的平移差约0.156mm，extent完全相同。这个差异与历史大坐标反算精度相容，但没有证据证明其原因；作为来源合同差异记录，不能说车型Mesh发生变化，也不能靠调整容差放行。即便忽略这个附加诊断，原15点NoRayHit已经足以停止。

原M8-F完整静态请求检查结果保留。它把自然下落与假设摆放姿态混在一起，本轮另列“相对真实已观测姿态的原支撑检查”以及独立原生XY/旋转漂移，明确剥离自然竖直下落；没有改地面、静态OBB、车道或接触阈值。

## 首次测量错误与保留记录

首次Spawn后、首Tick前 `get_transform()` 返回全零缓存，错误触发了265m/251m漂移判定。该次仅各运行1Tick，不能形成物理结论，全部记录保留在 `invalid_measurement_attempt`。修正为使用冻结Spawn RPC作漂移参考、首个有效snapshot作观测后，在完全相同两点重新测量一次。总Spawn调用4次：2次无效测量 + 2次有效Ego测量；零目标、零传感器、零图像。此为明确披露的仪器时序修复，非新位置/随机重试或在真实门禁失败后补拍。之后真正支撑门禁失败，没有再尝试。

## 独立状态与退出问题

A：两处Ego均真实落稳；实际最终Pose已记录，但没有创建相机，**相机实际位姿/传感器闭合NotRun**。

B：两处目标车辆 **NotRun**；Ego失败后未继续生成，不能拿Ego数量代替目标车辆成功数。

C：两处目标行人 **NotRun**；初始中心推导有预检记录，实际Bounding Box、脚部、身份未验证。

D：路灯视线警告没有取得新图像/实例解释，保持未验证。

E：唯一Shifted候选几何预检通过；真实移位、支撑和反事实身份全部NotRun。

F：六摄24路同步NotRun，关键帧0/8、RGB0/48、载荷0/192。单测中的时间戳负例不等于真实同步验证。

G：两处均未取得车辆+行人的自然场景支持，完整Smoke **NoGo**。

实际支持数分开记录：有效EgoSpawn2，静止Ego2，完整支撑Ego0；目标Spawn0、目标静止0、目标几何0、单视角身份0、多视角身份0、20–40m自然有效目标0。不是把NotRun目标算作检测漏检。全部DevelopmentOnly、HistoricalCoverageIncomplete、FinalEligible=false。

## 验证与架构决策

原水面、无地面、建筑、完整车身越界反例实测复核；NONE、树冠、静态冲突保留历史回放；多层和过期/缺失时间戳使用明确标注的合成单测。测试数量及失败记录见 `m8h1_test_results.json`。无阈值修补、无模型前向、无训练、无视觉驾驶。清理后恢复World设置并关闭本轮自有CARLA进程。

**停止进一步建设定制自动放置器。** 下一工作仅建议独立评审CARLA已有交通场景、ScenarioRunner或成熟标注数据，不自动启动M8-H2射线阈值修补。当前早停证明原几何证据合同不能为本次自然落稳提供合格证明；不等于证明原生场景物理不可行。
''')
 print('NoGo; genuine Ego stability yes, target/sensor downstream NotRun',flush=True)
if __name__=='__main__':main()
