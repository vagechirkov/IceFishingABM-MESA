import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from .belief import construct_dataset_info, generate_belief_matrix, generate_belief_mean_matrix
from .utils import x_y_to_i_j, find_peak
class ExplorationStrategy:
    def __init__(self, 
        grid_size: int = 100,
        ucb_beta=0.2, 
        tau=0.01):
    
        self.grid_size = grid_size
        self.ucb_beta = ucb_beta
        self.softmax_tau = tau
        self.mesh = np.array(np.meshgrid(range(self.grid_size), range(self.grid_size))).reshape(2, -1).T
        self.mesh_indices = np.arange(0, self.mesh.shape[0])
        self.belief_softmax = np.zeros((self.grid_size, self.grid_size))
        self.other_agent_locs = np.empty((0, 2))
        
        
    # Default algorithm selects destination randomly
    def choose_destination(self):
        self.destination = self.mesh[np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :]
    
    def movement_destination(self):
        self.choose_destination()


class GPExplorationStrategy(ExplorationStrategy):
    def __init__(self):
        super().__init__()
        self.social_gpr = GaussianProcessRegressor(kernel=RBF(12), random_state=0, optimizer=None)
        self.social_feature_m = np.zeros((self.grid_size, self.grid_size))
        self.social_feature_std = np.zeros((self.grid_size, self.grid_size))
        self.success_locs = np.empty((0, 2))
        self.success_gpr = GaussianProcessRegressor(kernel=RBF(5), random_state=0, optimizer=None)
        self.success_feature_m = np.zeros((self.grid_size, self.grid_size))
        self.success_feature_std = np.zeros((self.grid_size, self.grid_size))
        self.failure_locs = np.empty((0, 2))
        self.failure_gpr = GaussianProcessRegressor(kernel=RBF(5), random_state=0, optimizer=None)
        self.failure_feature_m = np.zeros((self.grid_size, self.grid_size))
        self.failure_feature_std = np.zeros((self.grid_size, self.grid_size))
        self.belief_m = np.zeros((self.grid_size, self.grid_size))
        self.belief_std = np.zeros((self.grid_size, self.grid_size))
        self.w_social = 0.4
        self.w_success = 0.3
        self.w_failure = 0.3

    def calculate_features(self, locations, gpr, grid_size):
        if locations.size == 0:
            feature_m = np.zeros((grid_size, grid_size))
            feature_std = np.zeros((grid_size, grid_size))
        else:
            gpr.fit(locations, np.ones(locations.shape[0]))
            feature_m, feature_std = generate_belief_mean_matrix(grid_size, gpr, return_std=True)
        return feature_m.T, feature_std.T

    def compute_beliefs(self):
        self.belief_m = self.w_social * self.social_feature_m + \
                              self.w_success * self.success_feature_m - \
                              self.w_failure * self.failure_feature_m

        self.belief_std = np.sqrt(
            self.w_social ** 2 * self.social_feature_std ** 2 +
            self.w_success ** 2 * self.success_feature_std ** 2 +
            self.w_failure ** 2 * self.failure_feature_std ** 2)

        self.belief_ucb = self.belief_m + self.ucb_beta * self.belief_std

        self.belief_softmax = np.exp(self.belief_ucb / self.softmax_tau) / np.sum(
            np.exp(self.belief_ucb / self.softmax_tau))

    def choose_destination(self):
        self.destination = self.mesh[np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :]

    def movement_destination(self):
        self.social_feature_m, self.social_feature_std = self.calculate_features(
            self.other_agent_locs, self.social_gpr, self.grid_size)

        if not self.agent._is_moving:
            self.success_feature_m, self.success_feature_std = self.calculate_features(
                self.success_locs, self.success_gpr, self.grid_size)

            self.failure_feature_m, self.failure_feature_std = self.calculate_features(
                self.failure_locs, self.failure_gpr, self.grid_size)

        self.compute_beliefs()
        self.choose_destination()
