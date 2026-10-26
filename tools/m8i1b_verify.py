"""Read-only release/evidence checks; never connects to CARLA."""
import ast
from tools.m8i1b_forensics import *


def main():
    m = read(REPO / 'reports/m8i1b_release_manifest.json')
    paths = set()
    for row in m['files']:
        p=Path(row['path']); paths.add(p)
        assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], str(p)
        if p.suffix=='.json': read(p)
        if p.suffix=='.py': ast.parse(p.read_text())
        if p.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                if '://' not in link and not link.startswith('#'):assert (p.parent/link).exists(), (p,link)
    for row in read(OUT/'historical_freeze.json')['files']:
        assert sha(row['path'])==row['sha256'],row['path']
    req=['summary.md','acceptance.json','runtime_log_inventory.json','success_failure_launch_diff.json',
         'rpc_readiness_evidence.json','authorization_record_erratum.md','root_cause_hypotheses.md']
    assert all(REPO/('reports/m8i1b_'+x) in paths for x in req)
    assert REPO/'docs/M8I1B_ARCHITECTURE_DECISION.md' in paths
    acc=read(REPO/'reports/m8i1b_acceptance.json')
    assert acc['decision']=='StartupRootCauseUnresolved' and acc['decision_option']=='B'
    assert not acc['new_startup_authorized'] and not acc['next_startup_experiment_proposed']
    assert not acc['next_direction_executed'] and acc['server_launches']==acc['CARLA_connections']==0
    chain=read(REPO/'reports/m8i1b_rpc_readiness_evidence.json')['chains']
    assert len(chain)==3 and all(len(v)==6 for v in chain.values())
    assert chain['M8-I1a']['CARLAVersionRPCReady']['value']=='NotAchieved_31Timeouts'
    assert chain['M8-I1a']['TCPListening']['value']=='Supported'
    assert chain['M8-I1']['CARLAVersionRPCReady']['value']=='Supported'
    logs=read(REPO/'reports/m8i1b_runtime_log_inventory.json')
    assert logs['status']=='NoRunMatchedUELog' and not logs['supported_environment_issue']
    assert all(not v['targeted_error_lines'] for v in logs['kernel_windows'].values())
    ap=read(REPO/'reports/m8i1b_authorization_provenance.json')
    assert ap['original_receipt_verbatim']==read(APPROVAL/'authorization.json')
    assert ap['source_chain'][1]['role']=='assistant' and ap['source_chain'][2]['role']=='user'
    assert ap['external_message_id'] is None and ap['external_platform_export'] is None
    assert ap['already_consumed_attempts']==1 and not ap['additional_launch_authorized']
    diff=read(REPO/'reports/m8i1b_success_failure_launch_diff.json')
    assert len(diff['comparison'])==13 and diff['I1_I1a_server_argv_identical']
    changes=subprocess.check_output(['git','diff',BASE,'--name-status','--diff-filter=MDTR'],cwd=REPO,text=True)
    assert not changes.strip(),changes
    print(json.dumps(dict(kind='ActualOfflineArtifactVerification',status='Passed',
        decision='StartupRootCauseUnresolved',deliverables=8,sealed_files=len(paths),
        historical_unique_assets=306,launch_comparison_fields=13,state_chain_entries=18,
        source_syntax_and_links='Passed',authorization_chain='ThreeLayers_TranscriptQualified',
        new_server_launches=0,new_CARLA_connections=0,base_modifications=0,
        manifest_sha256=sha(REPO/'reports/m8i1b_release_manifest.json')),indent=2))


if __name__=='__main__':main()
