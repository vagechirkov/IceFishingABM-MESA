import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from .belief import generate_belief_mean_matrix


### ALGORITHM 1: DEAFULT EXPLORATION STRATEGY

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
        self.destination = self.mesh[np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :]
        return self.destination

    def _check_input(self, input_data):
        assert input_data.ndim == 2, "Input data must have shape (n data points, 2)"
        assert input_data.shape[1] == 2, "Input data must have shape (n data points, 2)"



### ALGORITHM 2:  RANDOM WALKER EXPLORATION STRATEGY


class RandomWalkerExplorationStrategy(ExplorationStrategy):

    
    """
    Implementation of a Lévy random walk model for simulating ice angler movement patterns.
    This model incorporates social cues and environmental factors to simulate realistic
    fishing site selection behavior.

    The Lévy walk is characterized by a power-law distribution of step lengths, which produces
    a combination of short, localized movements and occasional long-distance relocations.
    This pattern has been observed in many foraging animals and has been shown to be an
    efficient search strategy when resources are sparsely and randomly distributed.

    Mathematical Model:
    ------------------
    The probability density function P(d) of step lengths d follows:
        P(d) = C * d^(-μ)
    where:
        - C is the normalization constant
        - μ (mu) is the Lévy exponent
        - d is the step length

    The cumulative distribution function is used for generating random steps:
        F(d) = C * (d^(1-μ) - dmin^(1-μ))/(1-μ)
    
    Parameters:
    -----------
    mu : float, default=1.5
        The Lévy exponent (μ) controlling the power-law distribution of step lengths.
        - Values between 1 and 3 produce superdiffusive behavior
        - μ ≈ 2.0 corresponds to optimal foraging in 2D spaces
        - Lower values increase the frequency of long jumps
        - Higher values make movement more Brownian-like

    dmin : float, default=1.0
        Minimum step length in the spatial units of the simulation grid.
        - Acts as a lower cutoff for the power-law distribution
        - Prevents singularity at d=0
        - Should be set based on minimum meaningful movement distance

    L : float, default=10.0
        Maximum step length allowed in the spatial units of the simulation grid.
        - Acts as an upper cutoff for the power-law distribution
        - Should be set based on physical constraints of the environment
        - Prevents unrealistic jumps across the entire space

    alpha : float, default=0.1
        Social influence parameter controlling the strength of attraction to other anglers.
        - Range: [0,1] where:
            0 = no social influence (pure Lévy walk)
            1 = strongest social influence
        - Determines probability of being attracted to nearby occupied sites
        - Higher values increase clustering behavior

    random_state : int, default=0
        Seed for random number generator to ensure reproducibility.

    Attributes:
    -----------
    C : float
        Normalization constant calculated as:
        C = (1-μ)/((L^(1-μ)) - (dmin^(1-μ)))
        Ensures the probability distribution integrates to 1

    Methods:
    --------
    _levy_flight()
        Generates a single Lévy flight step based on the power-law distribution.
    
    _adjust_for_social_cue(current_pos, social_locations)
        Modifies movement based on locations of other angents within social range.

    choose_destination(social_locs, resource_locs, obstacle_locs, social_cue=True)
        Determines next destination considering environmental and social factors.

    Example:
    --------
    
    """
    def __init__(self, mu: float = 1.5, dmin: float = 1.0, L: float = 10.0,
                 alpha: float = 0.1, random_state: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu                                                                                 # Levy exponent
        self.dmin = dmin
        self.L = L
        self.C = (1 - self.mu) / ((self.L ** (1 - self.mu)) - (self.dmin ** (1 - self.mu)))
        self.alpha = alpha
        self.random_state = random_state

    def choose_destination(self, social_locs, catch_locs, loss_locs, social_cue=False):
        """
        Choose destination based on social and private information
        """
        self._check_input(social_locs)
        self._check_input(catch_locs)
        self._check_input(loss_locs)
        self.destination = self._levy_flight()
        
        # Handle social cue if detected
        if social_cue:
            self.destination = self._adjust_for_social_cue(self.destination, social_locs)
        
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
        d = ((self.L**(1 - self.mu) - self.dmin**(1 - self.mu)) * u + self.dmin**(1 - self.mu))**(1 / (1 - self.mu))
        
        # Sample a random angle uniformly between 0 and 2π
        theta = np.random.uniform(0, 2 * np.pi)

        # Convert polar coordinates (d, θ) into Cartesian coordinates (dx, dy)
        dx = d * np.cos(theta)
        dy = d * np.sin(theta)

        # Assume current position is the center of the grid (grid_size / 2, grid_size / 2)
        current_pos = np.array([self.grid_size // 2, self.grid_size // 2])

        # Compute the new position and ensure it's within the grid boundaries
        
        new_position = np.clip(current_pos + np.array([dx, dy]), 0, self.grid_size - 1).astype(int)

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
            if np.random.rand() == prob_social:
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



###  ALGORITHM 3: GP EXPLORATION STRATEGY


class GPExplorationStrategy(ExplorationStrategy):
    def __init__(self, social_length_scale: float = 12, success_length_scale: float = 5,
                 failure_length_scale: float = 5, w_social: float = 0.4, w_success: float = 0.3, w_failure: float = 0.3,
                 random_state: int = 0, **kwargs):
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

    def _calculate_gp_feature(self, locations, gpr, grid_size):

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

        self.social_feature_m, self.social_feature_std = self._calculate_gp_feature(
            other_agent_locs, self.social_gpr, self.grid_size)

        self.success_feature_m, self.success_feature_std = self._calculate_gp_feature(
            success_locs, self.success_gpr, self.grid_size)

        self.failure_feature_m, self.failure_feature_std = self._calculate_gp_feature(
            failure_locs, self.failure_gpr, self.grid_size)

        self._compute_beliefs()

        self.destination = self.mesh[np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :]
        return self.destination
