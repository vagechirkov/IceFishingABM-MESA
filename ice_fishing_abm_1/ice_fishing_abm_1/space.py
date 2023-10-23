from mesa.space import MultiGrid


class MultiMicroMesoGrid(MultiGrid):

    def __init__(self, width: int, height: int, torus: bool, meso_scale_step: int = 10) -> None:
        super().__init__(width, height, torus)

        self.meso_scale_step = meso_scale_step
        assert width % meso_scale_step == 0, "width must be divisible by meso_scale_step"
        assert height % meso_scale_step == 0, "height must be divisible by meso_scale_step"
        self.meso_grid = MultiGrid(width // meso_scale_step, height // meso_scale_step, torus)
