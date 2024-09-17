from ice_fishing_abm_1.ice_fishing_abm_gp.patch_evaluation_subroutine import PatchEvaluationSubroutine


def test_patch_evaluation_subroutine_default():
    pes = PatchEvaluationSubroutine(threshold=5)

    assert pes.stay_on_patch(1, 10) is True
    assert pes.stay_on_patch(5, 10) is True
    assert pes.stay_on_patch(6, 10) is False


