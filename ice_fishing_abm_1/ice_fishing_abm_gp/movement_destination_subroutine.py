import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from .belief import generate_belief_mean_matrix


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
        self.destination = None

    def choose_destination(self, success_locs, failure_locs, other_agent_locs):
        """
        Select destination randomly
        """
        self._check_input(success_locs)
        self._check_input(failure_locs)
        self._check_input(other_agent_locs)
        raise NotImplementedError

        self.destination = self.mesh[np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :]
        return self.destination

    def _check_input(self, input_data):
        assert input_data.ndim == 2, "Input data must have shape (n data points, 2)"
        assert input_data.shape[1] == 2, "Input data must have shape (n data points, 2)"

    def movement_destination(self):
        self.choose_destination()


class GPExplorationStrategy(ExplorationStrategy):
    def __init__(self, social_length_scale=12, success_length_scale=5, failure_length_scale=5,
                 w_social=0.4, w_success=0.3, w_failure=0.3, random_state=0):
        super().__init__()
        # parameters for the Gaussian Process Regressors
        self.social_length_scale = social_length_scale
        self.success_length_scale = success_length_scale
        self.failure_length_scale = failure_length_scale
        self.w_social = w_social
        self.w_success = w_success
        self.w_failure = w_failure
        self.random_state = random_state

        # initialize Gaussian Process Regressors
        grid_shape = (self.grid_size, self.grid_size)
        # Social
        self.social_gpr = GaussianProcessRegressor(kernel=RBF(self.social_length_scale),
                                                   random_state=self.random_state, optimizer=None)
        self.social_feature_m = np.zeros(grid_shape)
        self.social_feature_std = np.zeros(grid_shape)
        # Success
        self.success_gpr = GaussianProcessRegressor(kernel=RBF(self.success_length_scale),
                                                    random_state=self.random_state, optimizer=None)
        self.success_feature_m = np.zeros(grid_shape)
        self.success_feature_std = np.zeros((self.grid_size, self.grid_size))
        # Failure
        self.failure_gpr = GaussianProcessRegressor(kernel=RBF(self.failure_length_scale),
                                                    random_state=self.random_state, optimizer=None)
        self.failure_feature_m = np.zeros(grid_shape)
        self.failure_feature_std = np.zeros(grid_shape)

        # initialize beliefs
        self.belief_ucb = None
        self.belief_m = np.zeros(grid_shape)
        self.belief_std = np.zeros(grid_shape)

    def calculate_features(self, locations, gpr, grid_size):
        if locations.size == 0:
            feature_m = np.zeros((grid_size, grid_size))
            feature_std = np.zeros((grid_size, grid_size))
        else:
            gpr.fit(locations, np.ones(locations.shape[0]))
            feature_m, feature_std = generate_belief_mean_matrix(grid_size, gpr, return_std=True)
        return feature_m.T, feature_std.T

    def _compute_beliefs(self):
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

    def choose_destination(self, success_locs, failure_locs, other_agent_locs):
        """
        Choose exploration destination using the GP model
        :param success_locs: shape (n data points, 2), locations of successful sampling attempts
        :param failure_locs: shape (n data points, 2), locations of failed sampling attempts
        :param other_agent_locs: shape (n data points, 2), locations of other agents
        :return: destination (x, y)
        """
        # make sure all inputs are in the correct format
        self._check_input(success_locs)
        self._check_input(failure_locs)
        self._check_input(other_agent_locs)

        self.social_feature_m, self.social_feature_std = self.calculate_features(
            other_agent_locs, self.social_gpr, self.grid_size)

        self.success_feature_m, self.success_feature_std = self.calculate_features(
            success_locs, self.success_gpr, self.grid_size)

        self.failure_feature_m, self.failure_feature_std = self.calculate_features(
            failure_locs, self.failure_gpr, self.grid_size)

        self._compute_beliefs()

        self.destination = self.mesh[np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :]
        return self.destination
