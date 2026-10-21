"""Explicit conservative public object result contract; hypotheses never trusted by default."""
import json
from pathlib import Path
from tools.m5l_common import write,asset

def main():
 records=[]
 for split in ['val','regression','test']:
  raw=json.loads(Path(f'runs/m5l_objects_{split}.json').read_text());n=len(raw['actors']);conditional=sum(r['valid']for r in raw['actors']);records.append(dict(split=split,actor_views=n,conditional_plane_estimates=conditional,public_trusted_estimates=0,public_status='Unknown_ContactAndPlaneUnverified',conditional_abstain=n-conditional,all_public_positions_abstain=True,reference=asset(f'runs/m5l_objects_{split}.json')))
 write('reports/m5l_contact_contract.json',dict(kind='explicit_research_output_contract',records=records,limits='locate() returns conditional hypothesis+reason; no officialinferenceorPlanneradapter registered. Numericconditionalerror isresearchfeasibility, notverifiedcontactlocalization. To liftUnknown requiresRGBcontact/occlusion andindependentplaneconfidence study. No GTgroundheight used.'))
if __name__=='__main__':main()
