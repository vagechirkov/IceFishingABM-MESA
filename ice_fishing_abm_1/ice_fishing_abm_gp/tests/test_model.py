import mesa
import pandas as pd
import pytest

from ice_fishing_abm_1.ice_fishing_abm_gp.movement_destination_subroutine import ExplorationStrategy
from ice_fishing_abm_1.ice_fishing_abm_gp.patch_evaluation_subroutine import PatchEvaluationSubroutine
from ice_fishing_abm_1.ice_fishing_abm_gp.model import Model


@pytest.mark.happy
def test_model():
    exploration_strategy = ExplorationStrategy()

    results = mesa.batch_run(
        Model,
        parameters={
            "exploration_strategy": exploration_strategy,
            "exploitation_strategy": [PatchEvaluationSubroutine(threshold=10), PatchEvaluationSubroutine(threshold=20)],
            "grid_size": 100,
            "number_of_agents": 5,
            "n_resource_clusters": 2,
            "resource_quality": 0.8,
            "resource_cluster_radius": 5,
            "keep_overall_abundance": True,
        },
        iterations=1,
        max_steps=10,
        data_collection_period=-1,  # only the last step
    )
    results = pd.DataFrame(results)
    assert results.loc[0, 'exploitation_strategy'].threshold == 10
    assert 'collected_resource' in results.columns
    assert 'is_sampling' in results.columns
    assert 'is_moving' in results.columns




