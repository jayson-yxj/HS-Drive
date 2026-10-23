"""Pinned official source and interface fragments; not a full framework smoke."""
import ast,math,xml.etree.ElementTree as ET,importlib.metadata as md
from enum import Enum
from pathlib import Path
from collections import Counter
import yaml
from tools.m7a_common import read,write,asset,provenance

ROOT=Path('reports/m7a_sources')

def main():
    commit=read(ROOT/'bench2drive_commit.json')['sha']
    path=ROOT/'leaderboard__leaderboard__autoagents__agent_wrapper.py.txt'
    tree=ast.parse(path.read_text());names={'QUALIFIER_SENSORS_LIMITS','SENSORS_LIMITS','ALLOWED_SENSORS'}
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_sensor_configuration']
    class Track(Enum):
        SENSORS='SENSORS';MAP='MAP';SENSORS_QUALIFIER='SENSORS_QUALIFIER';MAP_QUALIFIER='MAP_QUALIFIER'
    class Invalid(Exception):pass
    env=dict(math=math,Track=Track,SensorConfigurationInvalid=Invalid,MAX_ALLOWED_RADIUS_SENSOR=3.0)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    cfg=yaml.safe_load(Path('configs/cameras.yaml').read_text())
    sensors=[dict(type='sensor.camera.rgb',id=n,x=c['xyz_m'][0],y=c['xyz_m'][1],z=c['xyz_m'][2],
      yaw=c['yaw_deg'],pitch=c['pitch_deg'],roll=0,width=640,height=384,fov=100) for n,c in cfg['cameras'].items()]
    checks=[]
    for label,ss,track,expected in [('six_rgb_sensors',sensors,Track.SENSORS,True),
       ('six_rgb_qualifier',sensors,Track.SENSORS_QUALIFIER,False),
       ('duplicate_id',sensors+[sensors[0]],Track.SENSORS,False),
       ('oracle_map_sensor',sensors+[dict(type='sensor.opendrive_map',id='GTmap')],Track.SENSORS,False)]:
        try:env['validate_sensor_configuration'](ss,track,track.value);actual=True;reason=None
        except Invalid as e:actual=False;reason=str(e)
        checks.append(dict(test=label,accepted=actual,expected=expected,passed=actual==expected,reason=reason))
    dev=ET.parse(ROOT/'leaderboard__data__drivetransformer_bench2drive_dev10.xml.txt').getroot()
    routes=dev.findall('route')
    maps={p.stem for p in Path('/root/gpufree-data/simulators/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps').rglob('*.umap')}
    val=ROOT/'leaderboard__data__bench2drive_0.0.4_val.xml.txt'
    valinfo=dict(status='Unavailable_incomplete_source_transfer',route_count=None,unique_route_ids=None,required_maps=None)
    if val.exists():
        rr=ET.parse(val).getroot().findall('route')
        valinfo=dict(status='Parsed',route_count=len(rr),unique_route_ids=len({r.attrib['id'] for r in rr}),required_maps=sorted({r.attrib['town'] for r in rr}),
          scenarios=dict(Counter(s.attrib.get('type') for r in rr for s in r.findall('scenarios/scenario'))))
    deps={}
    for package in ['carla','numpy','networkx','Shapely','py-trees','opencv-python','dictor','ephem','xmlschema','pygame']:
        try:deps[package]=md.version(package)
        except md.PackageNotFoundError:deps[package]='NotInstalled'
    write('reports/m7a_bench2drive_feasibility.json',dict(kind='OfficialPinnedSourceReviewAndExtractedFunctionSmoke',provenance=provenance(),
      official_commit=commit,official_version='0.0.4',status='Partial_EnvironmentPreparationOnly_NoOfficialEvaluation',
      source='https://github.com/Thinklab-SJTU/Bench2Drive/tree/'+commit,
      leaderboard_scenario_runner='Vendored directories at same commit, no separate compatible release proven',
      version_mismatch='Top README CARLA0.9.15; bundled scenario_runner/CARLA_VER still0.9.13; evaluator minimum0.9.10.1 is not version equivalence',
      dependencies=dict(installed_SDK_environment=deps,official=['Python3.7/3.8 documented','numpy1.18.4','networkx2.2','Shapely1.7.1','py-trees0.8.3','opencv4.2.0.32'],decision='Separate compatible environment required; no downgrade of HSDrive'),
      agent_contract=dict(entry='AutonomousAgent(carla_host,carla_port,debug); sensors(); run_step(input_data,timestamp)->VehicleControl',
        sensor_payload='sensor ID -> (frame, data); camera BGRA array; wrapper requests same GameTime frame',
        six_camera_configuration='HS mounts valid in SENSORS max8/radius3m; qualifier max4 incompatible',
        image_period='HS historical10Hz sensor_tick .1 vs evaluator20Hz .05 exact-frame collection; must explicitly adapt/verify, not silently repeat old images',
        control='scenario_manager owns world.tick and ego.apply_control; cannot run alongside HSDrive Tick Owner',
        source_pose='Global CARLA actor pose prohibited as a visual model feature; calibrated mount and permitted sensors only'),
      extracted_function_smoke=checks,full_framework_import='NotRun_dependency_environment_not_installed',
      simulator_smoke='NotRun_no_new_world',complete_220='NotRun_no_qualified_visual_agent',official_score=None,
      routes=dict(v004=valinfo,dev10=dict(count=len(routes),towns=sorted({r.attrib['town'] for r in routes}),
        historical_development_only=True,not_claimed_subset_of_new_v004_until_ID_audit=True),
        installed_map_files=sorted(maps),installed_maps_loaded_this_stage=False,
        official_setup_requires_additional_maps=True,
        missing_dev10_maps=sorted({r.attrib['town'] for r in routes}-maps)),
      metrics=dict(driving_score='mean of score_route*product penalties over fixed220; min_speed penalty unused',
        success_rate='Completed or Perfect with no non-min-speed infractions /220',
        completeness='220 unique expected IDs, failures/crashes retained; merger count check alone cannot detect duplicate/missing ID substitution',
        missing_routes='official merger denominator remains220; do not omit failed cases',
        no_Oracle_scoring=True),
      resources=dict(gpu='RTX4090 24GB available',joint_model_CARLA_memory='NotRun',Town12_13_resource='NotMeasured',
        current_frozen_Mobile_offline_forward='reports/m7a_shadow_rgb_forward.json; not full B2D memory feasibility'),
      recommendation='Go only bounded isolated environment preparation; Dev10 development after readiness; full220 NoGo pending qualified visual closed loop',
      mature_camera3D_P0=dict(status='NotApproved_missing_new_checkpoint_and_4090_forward_evidence',
        inherited=asset('reports/m6a_perception_framework_feasibility.json'),official_asset_retried=False,
        no_new_custom_head=True,DirectCenterHead='CandidateNoGo_no_training_extension')))
    print('Official sensor-function checks',sum(c['passed'] for c in checks),'/',len(checks),'routes',valinfo)

if __name__=='__main__':main()
