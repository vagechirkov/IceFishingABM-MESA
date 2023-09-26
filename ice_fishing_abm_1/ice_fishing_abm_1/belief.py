import numpy as np


class Belief:
    def __init__(self, size: tuple[int, int]):
        self.belief_alpha = np.ones(shape=tuple(size), dtype=float)
        self.belief_beta = np.ones(shape=tuple(size), dtype=float)

        # # belief on the level of individual cells
        # self.micro_belief = np.zeros(shape=(model.grid.width, model.grid.height), dtype=float)
        #
        # # belief on the level of 10x10 cells (meso-scale)
        # assert model.grid.width % 10 == 0, 'grid width must be divisible by 10 to have a meso scale belief'
        # assert model.grid.height % 10 == 0, 'grid height must be divisible by 10 to have a meso scale belief'
        # self.meso_belief = np.zeros(shape=(model.grid.width // 10, model.grid.height // 10), dtype=float)
        #
        # # belief on the level of the whole grid (macro-scale)
        # self.macro_belief = 1e-6

    def update_belief(self, location: tuple[int, int], x: float):
        """
        x ∼ Bern(θ), θ ∼ beta(α, β), θ∈[0,1]

        posterior = beta(α_N, β_N), where α_N = α_N-1 + x and β_N = β_N-1 + 1 - x
        """
        alpha = self.belief_alpha[location]
        beta = self.belief_beta[location]

        self.belief_alpha[location] = alpha + x
        self.belief_beta[location] = beta + 1 - x

    def get_catch_mean(self):
        return self.belief_alpha / (self.belief_alpha + self.belief_beta)





