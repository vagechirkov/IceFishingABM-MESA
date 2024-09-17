class PatchEvaluationSubroutine:
    def __init__(self, threshold, **kwargs):
        # TODO: implement parameters
        self.threshold = threshold

    def stay_on_patch(self, time_on_patch, time_since_last_catch):
        if (time_on_patch > self.threshold) and (time_since_last_catch > self.threshold):
            return False
        return True


class GPPatchEvaluationSubroutine(PatchEvaluationSubroutine):
    def __init__(self, threshold, **kwargs):
        super().__init__(threshold)
        pass

    def stay_on_patch(self, time_on_patch, time_since_last_catch):
        return super().stay_on_patch(time_on_patch, time_since_last_catch)
