import numpy as np
from .utils import x_y_to_i_j
from .belief import generate_belief_mean_matrix

class GPMovementDestinationSubroutine:
    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model
        self.mesh = agent.mesh
        self.mesh_indices = agent.mesh_indices
        self.ucb_beta = agent.ucb_beta
        self.softmax_tau = agent.softmax_tau

    def calculate_features(self, locations, gpr, grid_size):
        if locations.size == 0:
            feature_m = np.zeros((grid_size, grid_size))
            feature_std = np.zeros((grid_size, grid_size))
        else:
            gpr.fit(locations, np.ones(locations.shape[0]))
            feature_m, feature_std = generate_belief_mean_matrix(grid_size, gpr, return_std=True)
        return feature_m.T, feature_std.T

    def compute_beliefs(self):
        self.agent.belief_m = self.model.w_social * self.agent.social_feature_m + \
                              self.model.w_success * self.agent.success_feature_m - \
                              self.model.w_failure * self.agent.failure_feature_m

        self.agent.belief_std = np.sqrt(
            self.model.w_social ** 2 * self.agent.social_feature_std ** 2 +
            self.model.w_success ** 2 * self.agent.success_feature_std ** 2 +
            self.model.w_failure ** 2 * self.agent.failure_feature_std ** 2)

        self.agent.belief_ucb = self.agent.belief_m + self.ucb_beta * self.agent.belief_std

        self.agent.belief_softmax = np.exp(self.agent.belief_ucb / self.softmax_tau) / np.sum(
            np.exp(self.agent.belief_ucb / self.softmax_tau))

    def choose_destination(self):
        self.agent._destination = self.mesh[np.random.choice(self.mesh_indices, p=self.agent.belief_softmax.reshape(-1)), :]

    def movement_destination(self):
        self.agent.social_feature_m, self.agent.social_feature_std = self.calculate_features(
            self.agent.other_agent_locs, self.agent.social_gpr, self.model.grid_size)

        if not self.agent._is_moving:
            self.agent.success_feature_m, self.agent.success_feature_std = self.calculate_features(
                self.agent.success_locs, self.agent.success_gpr, self.model.grid_size)

            self.agent.failure_feature_m, self.agent.failure_feature_std = self.calculate_features(
                self.agent.failure_locs, self.agent.failure_gpr, self.model.grid_size)

        self.compute_beliefs()
        self.choose_destination()
