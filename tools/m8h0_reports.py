"""Architecture review decisions with independent source/geometry/history/runtime states."""
import json
from pathlib import Path
from tools.m8h0_common import *
def read(n):return json.loads(Path('reports/m8h0_'+n+'.json').read_text())
def md(p,s):
 with Path(p).open('x') as f:f.write(s)
def main():
 v=read('native_vehicle_spawn_inventory');p=read('pedestrian_navigation_inventory');t=read('curated_template_candidates');assert len(t['templates'])==4
 write('reports/m8h0_acceptance.json',dict(**provenance('ArchitectureReviewOnly_ActualReadOnlyServerEvidence'),overall='NativeCandidateSourceReviewed',architecture_choice='C_NativeSourcesPlusCuratedFixedTemplates',template_candidates='CuratedTemplateCandidateGo',complete_geometry='TemplateGeometryPartial',historical_independence='HistoricalCoverageIncomplete',next_stage_proposal='DevelopmentSmokeProposalEligible',proposal_not_execution_authorization=True,actual_Actor_Smoke='NotRun',render_identity='NotRun',physical_contact='NotRun',training='NotAuthorized',visual_control='NotAuthorized',static_facilities='StaticFacilityPlacementNotReady',native_vehicle_points=v['count'],vehicle_proxy_supported=v['geometry_counts']['GeometryCandidateProxySupported'],navigation_requests=p['count'],person_proxy_supported=p['geometry_counts']['GeometryCandidateProxySupported'],frozen_templates=4,preferred_next_proposal_templates=['m8h0_template_00','m8h0_template_02'],why_preferred='123.24m Ego separation and no conservative static-OBB warning on either pedestrian view; vehicle lamp-box warnings remain, no replacement or extra search',M8G_preserved='Partial / PlacementFeasibilityNoGo / TrainingReadinessNoGo',base_files_unchanged=True,known_limits=['One map, one vehicle and one pedestrian blueprint','Only nominal 20-40m pinhole frustum; actual visibility unverified','Every template has at least one possible static OBB occluder','No initial-height transform is approved as final settled Actor pose','No same-candidate randomized source comparison; source superiority not proven','No independent split or new Final','Shifted transforms not allocated'],default_system='Unchanged',new_Final=0))
 rows=[]
 for x in t['templates']:
  va=x['nominal_views']['vehicle'];pa=x['nominal_views']['person'];cams=lambda q:'/'.join(v['camera'] for v in q['views'] if v['center_in_frustum'])
  rows.append(f"| {x['id'][-2:]} | {x['ego']['id']} | {x['vehicle']['id']} / {va['planar_distance_m']:.2f}m / {cams(va)} | {x['person']['id']} / {pa['planar_distance_m']:.2f}m / {cams(pa)} |")
 table='\n'.join(rows)
 md('reports/m8h0_summary.md',f'''# M8-H0 — CARLA Native Spawn & Curated Scenario Architecture Review

决策：选择 **C：原生候选来源 + 明确审核的固定模板**，限定为 Development 研究。状态为 `NativeCandidateSourceReviewed / CuratedTemplateCandidateGo / TemplateGeometryPartial / HistoricalCoverageIncomplete / DevelopmentSmokeProposalEligible`。

这只允许提出下一阶段 Smoke 申请，不是执行批准。**本轮新 Actor、传感器、RGB、GT 数据集、PETR 前向、训练均为 0。** 默认 Planner、Safety、Controller、Physics 和模型未改；M8-G 保持 `Partial / PlacementFeasibilityNoGo / TrainingReadinessNoGo`。

基础提交 `{BASE}`；分支 `feat/m8h0-native-scenario-source-review`。原始几何证据位于 `{ROOT}`，配置与报告逐文件带源码 SHA，最终 commit 封存 release manifest。

## 实际查询与来源资格

CARLA 0.9.15，Town02：只调用一次 `get_spawn_points()`，完整保存 **101 个**推荐点；导航接口恰好 **60 次**，60 次均返回坐标，按请求顺序保存；未追加随机查询。原生点完整 Transform、Road/Lane/Section/S、每点 15 条射线/车道检查、静态竞争 OBB、候选完整足迹、历史预检和六摄名义视锥支持全部可追溯。

| 来源 | 实际数 | 几何代理支持 | 未通过/未闭合 |
|---|---:|---:|---:|
| 地图车辆推荐点 | 101 | 42 | 59 |
| 服务器导航返回 | 60 | 28 | 32 |

车辆原因可重叠：Footprint 39、Covered 22、Static OBB 25、原生朝向与最近车道不一致 2。行人：Footprint 24、缺地面支持 6、Static OBB 8、Covered 1。全部保留，没有把原生来源当作接受规则。未测量其他车型尺寸：本轮仅复用历史 Tesla Model3 与 walker.0002 已记录资产尺寸；其他可用 Blueprint 名称不代表尺寸/支撑合格。

车辆原生 Z 全部比查询地图高度高 **0.5m**，初始 OBB 底部到射线地面约 **0.272–0.275m**。导航位置比参考地图面高约 **0.326–0.409m**；若误用为 walker Actor 中心，底部将进入支撑面约 **0.874–0.937m**。因此两个来源的初始位置都不等于最终落地状态。几何审核中的竖直对齐仅是假设姿态，保持原 XY/yaw、没有新 Actor 或 Physics 操作；后续必须用实际稳定 Actor 状态复核。

CARLA 官方文档明确车辆推荐点有初始抬高，导航返回是人行道上的目的地且不考虑已有 Actor；不能推出自然落地和无占用。[CARLA 0.9.15 API](https://carla.readthedocs.io/en/0.9.15/python_api/)

## 四个固定模板

最多四个模板的上限及选择规则先声明：仅从上述完整清单选，目标在 20–40m、按离 30m 的距离排序，每 Ego 固定一个车辆和一个行人，模板中心至少 50m，禁止结果不佳后补抽点。50m 只是 Development 调试布局间隔，**不是降低历史独立分组标准**。

| 模板 | Ego 原生 ID | 车辆 / 距离 / 相机方向 | 行人 / 距离 / 相机方向 |
|---|---|---|---|
{table}

四组共 12 对 Ego/目标间完整 OBB 包络关系均满足预声明间距，车辆和行人位置分别通过原地面语义、层面、0.10m inset、15 点支撑及静态 OBB 规则。已用冻结 M8-F 实现回放同一批存储查询，确认没有新增接受覆盖；额外原因仅补充诊断，不更改门禁。

模板先冻结，再做静态 OBB 视线审查，未按结果换样：四组均有至少一个路灯/站亭 OBB 可能遮挡目标中心。它们是近似几何警告，不是精确 Mesh 遮挡真值，也不能忽略。名义覆盖 front、rear、rear_left、rear_right；没有 front_left/front_right 的模板支持证据。六摄方向不能当成六个独立场景。

精确地图绑定包括 API OpenDRIVE SHA、地图包、BuiltData、导航二进制、服务器二进制 SHA 和本次环境库存 SHA；并不声称已逐个认证所有引用 Mesh。只读地面证据、完整足迹和历史隔离要求全部保留。Cone/Barrier 均为 `StaticFacilityPlacementNotReady`，没有硬凑目标。

## 历史与比较边界

原 M8-G 104 项封存资产及 **6,640 个历史源文件**校验；继续使用 **16,562 条**空间记录，所有候选在模板选择前完成足迹预检。5 个历史回归通过，包括 **3.635626m、4.595351m**。已恢复 OBB 用 XY polygon 距离，其余圆包络仅提供保守下界；历史 0.9.15/0.9.16 同名地图不假设 Mesh 等价。

四个模板的目标都有历史近邻，40m 观察包络与已评价场景重叠，车辆/行人 Blueprint 和布局家族重复。模板 01 离旧 Development01 Ego 约 1.17m，明确属于历史区域复用。全部仅 Development；没有新独立 Validation/Calibration/Final，旧 Held-out 不改名。

M8-G 的车辆 3/20 是两 phase、两个旧邻域结果；本轮 42/101 是 Town02 全原生清单的单姿态结果，不能直接计算因果改善或稳定性显著性。旧 Development00 周围 35m 的原生点 **0/5** 通过，Development01 为 **5/7**；切换来源没有自动解决旧困难区域。

## 对退出问题的回答

A：原生来源提供更多、身份明确且可复放的车辆候选，本地图一次清单有 42 个几何支持；**尚不能证明比旧策略更稳定**，范围和 phase 不匹配，旧00仍失败。

B：可以提供有来源、可审核的位置；60 个返回全部保存，28 个几何支持。服务器随机结果不能只靠客户端 seed 复现，必须回放已保存坐标。导航可达不等于生成、碰撞安全或渲染身份。

C：**候选层面可以**，四个模板通过未放宽的支撑/区域/间距代理；完整自然场景仍是 Partial，视线遮挡、实际落稳与身份未测。

D：未验证真实稳定姿态/接触、最终车身支撑、行人原点与生成高度、完整 Mesh 占用与遮挡、实例身份、多视角关联、传感器同步、出现/移位/移除以及新场景负例。包围盒空隙和看不到目标都不能当安全 Free。

E：历史覆盖缺口、重叠观察区域、同地图版本不确定性和同模板布局相关性阻止独立 Final、分布泛化和训练资格声明。

F：**值得申请一次有明确退出条件的两处 Development 真实 Actor Smoke**，优先模板00、02（Ego相距123.24m，行人方向无静态 OBB 视线警告，车辆路灯 OBB 警告仍待实测）。该建议不是本轮批准；不得自动运行。若真实 Actor/身份仍失败，停止专用放置器扩展，评估已有交通场景、ScenarioRunner 或现成标注数据，不继续射线阈值补丁。

验收 JSON、完整测试与日志、三幅几何诊断图、复现命令随附。源码/报告是工程交付，架构选择成功不代表场景生成或训练资格。
''')
 md('reports/m8h0_architecture_comparison.md','''# M8-H0 三方案统一比较

选择 **C：原生来源 + 少量固定审核模板**，理由是来源身份和失败边界清楚、候选精确可回放、无需恢复专用位置搜索器。选择的是下一项有限验证路径，不是已完成自然场景。

| 维度 | A：封存 M8-G 混合放置器 | B：原生 Spawn + Navigation | C：原生来源 + 审核模板 |
|---|---|---|---|
| 来源数量 | 80 预设候选；28 两 phase 代理支持 | 101 Spawn，60 导航；42/28 单姿态代理支持 | 4 固定候选组，每组 Ego+Vehicle+Person |
| 同组车/人 | 旧00车辆缺失，核心失败 | 存在36个 Ego 具备两类20–40m候选选项；仅离线有限清单配对 | 4组间距/区域/地面代理支持，真实状态未测 |
| 20–40m/相机 | 旧候选几何，无新RGB | 只计算六摄名义视锥，无遮挡认证 | 四组 front/rear/rear_left/rear_right；其他方向未覆盖 |
| 地面盲点 | overlap触发体、树木大OBB、Any查询歧义 | 同样保留这些未知；另有初始Z与行人位置/中心语义 | 精确固定候选逐项说明未知，不能消除Mesh/真实落稳缺口 |
| 新几何复杂度 | 继续修补成本高，本轮停止 | 复用原规则，无阈值放宽；来源解释较清楚 | 增加审核清单/姿态版本管理，复杂度有限 |
| 可复现性 | 记录可回放，搜索器难稳定通过 | Spawn绑定地图版本；导航必须保存真实返回坐标 | 精确Map/ID/Transform/资产/历史关系冻结 |
| 独立性 | HistoricalCoverageIncomplete | 同地图/历史近邻，不能把随机点称独立 | 全部Development，模板重复相关性显式保留 |
| 背景标注 | 库存OBB，身份局限未解 | 只读库存完整来源，未采集语义/Depth/Instance | 视线OBB警告逐项保留，实际背景身份NotRun |
| 维护与基准复用 | 离线诊断参考有价值，不继续扩展 | API简单，但直接自动采集风险大 | 推荐：绑定现有六摄K/T、ActorBoxCenter、身份与历史管线 |

不能用 42/101 对 3/20 宣称单变量性能提高：Town02空间范围、姿态阶段、取样分布不同，样本相关且未自然落稳。旧00附近原生车辆0/5这一反例仍保留。

后续申请仅两模板00/02，先检查生成和稳定状态，再检查身份与反事实；不得以已有架构报告代替实测。模板01/03作为本次完整审查记录保留，不能在失败后静默替换样本。所有模板均有视线OBB警告，需要真实图像诊断。此时无真实接触证明、训练许可或视觉控制许可。
''')
 print('architecture C; proposal eligible only; actual Smoke NotRun',flush=True)
if __name__=='__main__':main()
