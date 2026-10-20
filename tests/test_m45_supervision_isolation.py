import numpy as np
import pytest
from datasets.schema import validate_learned_inputs


@pytest.mark.parametrize('field',['occupancy_state','occupancy_valid','surface_semantic',
    'obstacle_mask','visibility_count','ray_conflict_mask','oracle_actors','oracle_map'])
def test_m45_privileged_fields_rejected_by_learned_contract(field):
    with pytest.raises(ValueError,match='Forbidden'):
        validate_learned_inputs({'images':np.zeros((6,3,4,4)),field:np.zeros((4,160,120))})
