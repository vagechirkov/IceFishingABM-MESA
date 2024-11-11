import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from .belief import generate_belief_mean_matrix


class ExplorationStrategy:
    """ Default exploration strategy class."""
    def __init__(self, grid_size: int = 100, ucb_beta=0.2, tau=0.01):
        self.grid_size = grid_size
        self.ucb_beta = ucb_beta
        self.softmax_tau = tau
        self.mesh = (
            np.array(np.meshgrid(range(self.grid_size), range(self.grid_size)))
            .reshape(2, -1)
            .T
        )
        self.mesh_indices = np.arange(0, self.mesh.shape[0])
        self.belief_softmax = np.random.uniform(0, 1, (self.grid_size, self.grid_size))
        self.belief_softmax /= np.sum(self.belief_softmax)  # Normalize the distribution
        self.other_agent_locs = np.empty((0, 2))
        self.destination = None

    def choose_destination(self, success_locs, failure_locs, other_agent_locs):
        """
        Select destination randomly
        """
        self._check_input(success_locs)
        self._check_input(failure_locs)
        self._check_input(other_agent_locs)
        self.other_agent_locs = other_agent_locs
        self.destination = self.mesh[
            np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :
        ]
        return self.destination

    def _check_input(self, input_data):
        assert input_data.ndim == 2, "Input data must have shape (n data points, 2)"
        assert input_data.shape[1] == 2, "Input data must have shape (n data points, 2)"


class RandomWalkerExplorationStrategy(ExplorationStrategy):
    """
    Do doc string here explaining main parameters of the model
    """

    def __init__(
        self,
        mu: float = 1.5,
        dmin: float = 1.0,
        L: float = 10.0,
        alpha: float = 0.1,
        random_state: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mu = mu  # Levy exponent
        self.dmin = dmin
        self.L = L
        self.C = (1 - self.mu) / (
            (self.L ** (1 - self.mu)) - (self.dmin ** (1 - self.mu))
        )
        self.alpha = alpha
        self.random_state = random_state

    def choose_destination(self, social_locs, catch_locs, loss_locs, social_cue=False):
        """
        Choose destination based on motion type (Brownian or Levy)
        """
        self._check_input(social_locs)
        self._check_input(catch_locs)
        self._check_input(loss_locs)
        self.destination = self._levy_flight()

        # Handle social cue if detected
        if social_cue:
            self.destination = self._adjust_for_social_cue(
                self.destination, social_locs
            )

        return self.destination

    def _levy_flight(self):
        """
        Sample a displacement distance using a Levy flight distribution and ensure
        the distance is within the given bounds (dmin and L).
        """
        # Sample from a uniform distribution
        u = np.random.uniform(0, 1)

        # Use inverse transform sampling to get d from P(d) = C d^(-mu)
        # Ensure the distance lies between dmin and L
        d = (
            (self.L ** (1 - self.mu) - self.dmin ** (1 - self.mu)) * u
            + self.dmin ** (1 - self.mu)
        ) ** (1 / (1 - self.mu))

        # Sample a random angle uniformly between 0 and 2π
        theta = np.random.uniform(0, 2 * np.pi)

        # Convert polar coordinates (d, θ) into Cartesian coordinates (dx, dy)
        dx = d * np.cos(theta)
        dy = d * np.sin(theta)

        # Assume current position is the center of the grid (grid_size / 2, grid_size / 2)
        current_pos = np.array([self.grid_size // 2, self.grid_size // 2])

        # Compute the new position and ensure it's within the grid boundaries

        new_position = np.clip(
            current_pos + np.array([dx, dy]), 0, self.grid_size - 1
        ).astype(int)

        return new_position

    def _adjust_for_social_cue(self, current_destination, social_locs):
        """
        Adjust destination based on social cues, if detected
        """
        if social_locs.size > 0:
            # Find the nearest social cue
            distances = np.linalg.norm(social_locs - current_destination, axis=1)
            nearest_social_loc = social_locs[np.argmin(distances)]

            # Compute probability to switch to the nearest social cue based on distance
            delta_d = distances.min()
            prob_social = np.exp(-self.alpha * delta_d)

            # With probability prob_social, move to the nearest social cue
            if np.random.rand() < prob_social:
                current_destination = nearest_social_loc

        return current_destination

    def _get_new_position(self, dx, dy):
        """
        Compute the new position from a displacement (dx, dy)
        """
        # Current position (we assume the walker starts from the center of the grid)
        current_position = np.array([self.grid_size // 2, self.grid_size // 2])

        # Calculate new position with periodic boundary conditions (wrap around the grid)
        new_x = (current_position[0] + dx) % self.grid_size
        new_y = (current_position[1] + dy) % self.grid_size

        return np.array([new_x, new_y])


class GPExplorationStrategy(ExplorationStrategy):
    def __init__(
        self,
        social_length_scale: float = 12,
        success_length_scale: float = 5,
        failure_length_scale: float = 5,
        w_social: float = 0.4,
        w_success: float = 0.3,
        w_failure: float = 0.3,
        random_state: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
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
        self.social_gpr = GaussianProcessRegressor(
            kernel=RBF(self.social_length_scale),
            random_state=self.random_state,
            optimizer=None,
        )
        self.social_feature_m = np.zeros(grid_shape)
        self.social_feature_std = np.zeros(grid_shape)
        # Success
        self.success_gpr = GaussianProcessRegressor(
            kernel=RBF(self.success_length_scale),
            random_state=self.random_state,
            optimizer=None,
        )
        self.success_feature_m = np.zeros(grid_shape)
        self.success_feature_std = np.zeros((self.grid_size, self.grid_size))
        # Failure
        self.failure_gpr = GaussianProcessRegressor(
            kernel=RBF(self.failure_length_scale),
            random_state=self.random_state,
            optimizer=None,
        )
        self.failure_feature_m = np.zeros(grid_shape)
        self.failure_feature_std = np.zeros(grid_shape)

        # initialize beliefs
        self.belief_ucb = None
        self.belief_m = np.zeros(grid_shape)
        self.belief_std = np.zeros(grid_shape)

    def _calculate_gp_feature(self, locations, gpr, grid_size):

        if locations.size == 0:
            feature_m = np.zeros((grid_size, grid_size))
            feature_std = np.zeros((grid_size, grid_size))
        else:
            gpr.fit(locations, np.ones(locations.shape[0]))
            feature_m, feature_std = generate_belief_mean_matrix(
                grid_size, gpr, return_std=True
            )
        return feature_m.T, feature_std.T

    def _compute_beliefs(self):
        self.belief_m = (
            self.w_social * self.social_feature_m
            + self.w_success * self.success_feature_m
            - self.w_failure * self.failure_feature_m
        )

        self.belief_std = np.sqrt(
            self.w_social**2 * self.social_feature_std**2
            + self.w_success**2 * self.success_feature_std**2
            + self.w_failure**2 * self.failure_feature_std**2
        )

        self.belief_ucb = self.belief_m + self.ucb_beta * self.belief_std

        self.belief_softmax = np.exp(self.belief_ucb / self.softmax_tau) / np.sum(
            np.exp(self.belief_ucb / self.softmax_tau)
        )

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

        self.social_feature_m, self.social_feature_std = self._calculate_gp_feature(
            other_agent_locs, self.social_gpr, self.grid_size
        )

        self.success_feature_m, self.success_feature_std = self._calculate_gp_feature(
            success_locs, self.success_gpr, self.grid_size
        )

        self.failure_feature_m, self.failure_feature_std = self._calculate_gp_feature(
            failure_locs, self.failure_gpr, self.grid_size
        )

        self._compute_beliefs()

        self.destination = self.mesh[
            np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :
        ]
        return self.destination
