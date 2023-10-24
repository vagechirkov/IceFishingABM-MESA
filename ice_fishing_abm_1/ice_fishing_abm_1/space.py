from mesa.space import MultiGrid


class MultiMicroMesoGrid(MultiGrid):

    def __init__(self, width: int, height: int, torus: bool, meso_scale_step: int = 10) -> None:
        super().__init__(width, height, torus)

        self.meso_scale_step = meso_scale_step
        assert width % meso_scale_step == 0, "width must be divisible by meso_scale_step"
        assert height % meso_scale_step == 0, "height must be divisible by meso_scale_step"
        self.meso_width = width // meso_scale_step
        self.meso_height = height // meso_scale_step

    def meso_coordinate(self, x: int, y: int) -> tuple[int, int]:
        """
        Return the meso grid coordinates for the given micro grid coordinates.
        """
        return x // self.meso_scale_step, y // self.meso_scale_step

    def micro_slice_from_meso_coordinate(self, x: int, y: int) -> tuple[slice, slice]:
        """
        Return the micro grid slice for the given meso grid coordinates.
        """
        return slice(x * self.meso_scale_step, (x + 1) * self.meso_scale_step), \
            slice(y * self.meso_scale_step, (y + 1) * self.meso_scale_step)
