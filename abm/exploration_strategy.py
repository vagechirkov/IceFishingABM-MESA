import numpy as np
from scipy.special import softmax
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KernelDensity

from abm.belief import generate_belief_mean_matrix


### ALGORITHM 1: DEAFULT EXPLORATION STRATEGY
class ExplorationStrategy:
    def __init__(self, grid_size: int = 100, ucb_beta=0.2, tau=0.01, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

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

    def choose_destination(
        self, *,
        current_position=np.empty((0, 2)),
        success_locs=np.empty((0, 2)),
        failure_locs=np.empty((0, 2)),
        other_agent_locs=np.empty((0, 2)),
    ):
        """
        Select destination randomly
        """
        assert (
            len(current_position.shape) == 1 and current_position.shape[0] == 2
        ), "Current position must be a 1D array with 2 elements"
        self._check_input(success_locs)
        self._check_input(failure_locs)
        self._check_input(other_agent_locs)
        self.other_agent_locs = other_agent_locs
        self.destination = self.mesh[
            np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :
        ]
        return self.destination

    def _check_input(self, input_data):
        if input_data.size > 0:
            assert input_data.ndim == 2, "Input data must have shape (n data points, 2)"
            assert (
                input_data.shape[1] == 2
            ), "Input data must have shape (n data points, 2)"

    @staticmethod
    def _zscore(M):
        mu = M.mean()
        sd = M.std()
        return (M - mu) / sd if sd > 0 else (M - mu)

    @staticmethod
    def _softmax(logits_flat, tau=1.0):
        probs = softmax(logits_flat / max(tau, 1e-8))
        if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
            probs = np.full_like(probs, 1.0 / probs.size)
        return probs


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

    choose_destination(social_locs, resource_locs, obstacle_locs)
        Determines next destination considering environmental and social factors.

    Example:
    --------

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
        self._prob_social = 0

    def choose_destination(
        self, *,
        current_position=np.empty((0, 2)),
        success_locs=np.empty((0, 2)),
        failure_locs=np.empty((0, 2)),
        other_agent_locs=np.empty((0, 2)),
    ):
        """
        Choose destination based on social and private information
        """
        current_position = np.array(current_position, dtype=np.int32)
        # Added assertions as later needed for testing
        assert (
            len(current_position.shape) == 1 and current_position.shape[0] == 2
        ), "Current position must be a 1D array with 2 elements"
        assert (
            success_locs.shape[1] == 2 if success_locs.size > 0 else True
        ), "Catch locations must be Nx2 array"
        assert (
            other_agent_locs.shape[1] == 2 if other_agent_locs.size > 0 else True
        ), "Failure locations must be Nx2 array"
        self._check_input(other_agent_locs)
        self._check_input(success_locs)
        self._check_input(failure_locs)
        _ = self._levy_flight(current_position)

        # Handle social cue if detected
        # if social_cue:
        _ = self._adjust_for_social_cue(current_position, other_agent_locs)

        return self.destination

    def _levy_flight(self, current_position):
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

        # Ensure current position data type is integer
        current_position = np.array(current_position, dtype=np.int32)

        new_x = int(np.clip(np.round(current_position[0] + dx), 0, self.grid_size - 1))
        new_y = int(np.clip(np.round(current_position[1] + dy), 0, self.grid_size - 1))

        self.destination = np.array([new_x, new_y], dtype=int)
        return self.destination

    def _adjust_for_social_cue(self, position, social_locs):
        """
        Adjust destination based on social cues, if detected
        """
        if social_locs.size > 0:
            # Find the nearest social cue
            distances = np.linalg.norm(social_locs - position, axis=1)
            nearest_social_loc = social_locs[np.argmin(distances)]

            # Compute probability to switch to the nearest social cue based on distance
            delta_d = distances.min()
            self._prob_social = np.exp(-self.alpha * delta_d)

            # MADE CONSISTENT WITH PAPER . DJ. 3.3.25. 
            '''
            We simulated the model for three
            levels of social selectivity (α = 10−5, , 10−2 , 1 )
            that correspond with high, intermediate, and no social learning, respectively.
            '''
            if np.random.rand() < self._prob_social: # FIXED 
                self.destination = np.array(nearest_social_loc, dtype=int)

        return self.destination


###  ALGORITHM 3: GP EXPLORATION STRATEGY
class KernelBeliefExploration(ExplorationStrategy):
    def __init__(
        self,
        social_length_scale: float = 12,
        success_length_scale: float = 5,
        failure_length_scale: float = 5,
        w_social: float = 0.25,
        w_success: float = 0.25,
        w_failure: float = 0.25,
        w_locality: float = 0.25,
        model_type="kde",
        normalize_features=False,
        w_as_attention_shares=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # parameters for the Gaussian Process Regressors
        self.social_length_scale = social_length_scale
        self.success_length_scale = success_length_scale
        self.failure_length_scale = failure_length_scale
        self.w_social = w_social
        self.w_success = w_success
        self.w_failure = w_failure
        self.w_locality = w_locality
        self.model_type = model_type.lower()
        self.normalize_features = normalize_features
        if w_as_attention_shares:
            assert np.allclose(np.sum([self.w_social, self.w_failure, self.w_success, self.w_locality]), 1.0)

        # initialize Gaussian Process Regressors
        grid_shape = (self.grid_size, self.grid_size)
        self.locality_feature = np.zeros(grid_shape)

        if self.model_type == "ucb":
            # Social
            self.social_gpr = GaussianProcessRegressor(
                kernel=RBF(self.social_length_scale),
                random_state=None,
                optimizer=None,
            )
            self.social_feature_m = np.zeros(grid_shape)
            self.social_feature_std = np.zeros(grid_shape)
            # Success
            self.success_gpr = GaussianProcessRegressor(
                kernel=RBF(self.success_length_scale),
                random_state=None,
                optimizer=None,
            )
            self.success_feature_m = np.zeros(grid_shape)
            self.success_feature_std = np.zeros((self.grid_size, self.grid_size))
            # Failure
            self.failure_gpr = GaussianProcessRegressor(
                kernel=RBF(self.failure_length_scale),
                random_state=None,
                optimizer=None,
            )
            self.failure_feature_m = np.zeros(grid_shape)
            self.failure_feature_std = np.zeros(grid_shape)
        elif self.model_type == "kde":
            self.social_feature_kde = np.zeros(grid_shape)
            self.success_feature_kde = np.zeros(grid_shape)
            self.failure_feature_kde = np.zeros(grid_shape)
        else:
            raise NotImplementedError

    def choose_destination(
        self,
        *,
        current_position=np.empty((0, 2)),
        success_locs=np.empty((0, 2)),
        failure_locs=np.empty((0, 2)),
        other_agent_locs=np.empty((0, 2)),
    ):
        """Choose an exploration destination based on KDE or GP-UCB beliefs."""

        self.update_features(
            current_position=current_position,
            success_locs=success_locs,
            failure_locs=failure_locs,
            other_agent_locs=other_agent_locs
        )

        if self.model_type == "kde":
            belief = self._compute_kde_beliefs()
        elif self.model_type == "ucb":
            belief = self._compute_ucb_beliefs()
        else:
            raise NotImplementedError

        self.belief_softmax = self._softmax(belief, tau=self.softmax_tau)
        idx = self.rng.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1))
        self.destination = self.mesh[idx, :]
        return self.destination

    def update_features(
        self,
        current_position,
        success_locs=np.empty((0, 2)),
        failure_locs=np.empty((0, 2)),
        other_agent_locs=np.empty((0, 2)),
    ):
        # make sure all inputs are in the correct format
        self._check_input(success_locs)
        self._check_input(failure_locs)
        self._check_input(other_agent_locs)

        self._compute_locality_feature(np.array(current_position))

        if self.model_type == "kde":
            self._compute_kde_features(
                other_agent_locs, success_locs, failure_locs
            )
        elif self.model_type == "ucb":
            self._compute_ucb_features(
                other_agent_locs, success_locs, failure_locs
            )
        else:
            raise NotImplementedError

    def _compute_locality_feature(self, current_position):
        # compute inverse distance from current_position (x, y) to all grid points

        # Epsilon to prevent division by zero
        epsilon = 1e-8

        # Create coordinate grids
        y_indices, x_indices = np.indices(self.locality_feature.shape)

        # Extract the current x and y coordinates
        current_x, current_y = current_position.flatten()

        # Calculate the Euclidean distance
        distance = np.sqrt((x_indices - current_x)**2 + (y_indices - current_y)**2)
        max_distance = np.max(distance)
        normalized_distance = distance / (max_distance + epsilon)
        self.locality_feature = (1.0 - normalized_distance).T

        if self.normalize_features:
            self.locality_feature = self._zscore(self.locality_feature)

    def _compute_ucb_features(self, _social_locs, _success_locs, _failure_locs):
        self.social_feature_m, self.social_feature_std = self._calculate_gp_feature(
            _social_locs, self.social_gpr, self.grid_size
        )

        self.success_feature_m, self.success_feature_std = self._calculate_gp_feature(
            _success_locs, self.success_gpr, self.grid_size
        )

        self.failure_feature_m, self.failure_feature_std = self._calculate_gp_feature(
            _failure_locs, self.failure_gpr, self.grid_size
        )

        # optional per-feature standardization
        if self.normalize_features:
            self.social_feature_m  = self._zscore(self.social_feature_m)
            self.success_feature_m = self._zscore(self.success_feature_m)
            self.failure_feature_m = self._zscore(self.failure_feature_m)

    def _compute_ucb_beliefs(self):
        self.belief_m = (
                self.w_social * self.social_feature_m
                + self.w_success * self.success_feature_m
                + self.w_failure * self.failure_feature_m
                + self.w_locality * self.locality_feature
        )

        self.belief_std = np.sqrt(
            self.w_social**2 * self.social_feature_std**2
            + self.w_success**2 * self.success_feature_std**2
            + self.w_failure**2 * self.failure_feature_std**2
        )

        return self.belief_m + self.ucb_beta * self.belief_std

    def _calculate_gp_feature(self, locations, gpr, grid_size):
        if locations.size == 0:
            feature_m = np.zeros((grid_size, grid_size))
            feature_std = np.zeros((grid_size, grid_size))
            return feature_m.T, feature_std.T

        gpr.fit(locations, np.ones(locations.shape[0]))
        feature_m, feature_std = gpr.predict(self.mesh, return_std=True)

        feature_m  = feature_m.reshape(grid_size, grid_size)
        feature_std =  feature_std.reshape(grid_size, grid_size)

        return feature_m.T, feature_std.T

    def _compute_kde_features(self, _social_locs, _success_locs, _failure_locs):
        self.social_feature_kde  = self._calculate_kde_feature(_social_locs,  self.social_length_scale)
        self.success_feature_kde = self._calculate_kde_feature(_success_locs, self.success_length_scale)
        self.failure_feature_kde = self._calculate_kde_feature(_failure_locs, self.failure_length_scale)

        if self.normalize_features:
            self.social_feature_kde  = self._zscore(self.social_feature_kde)
            self.success_feature_kde = self._zscore(self.success_feature_kde)
            self.failure_feature_kde = self._zscore(self.failure_feature_kde)

    def _compute_kde_beliefs(self):
        belief = (
                self.w_social * self.social_feature_kde
                + self.w_success * self.success_feature_kde
                - self.w_failure * self.failure_feature_kde
                + self.w_locality * self.locality_feature
        )
        return belief

    def _calculate_kde_feature(self, locations, length_scale):
        if locations.size == 0:
            return np.zeros((self.grid_size, self.grid_size))

        kde = KernelDensity(kernel="gaussian", bandwidth=length_scale)
        kde.fit(locations)
        log_dens = kde.score_samples(self.mesh)
        return np.exp(log_dens).reshape(self.grid_size, self.grid_size).T


# ALGORITHM 4: SOCIAL INFOTAXIS EXPLORATION STRATEGY
class SocialInfotaxisExplorationStrategy(ExplorationStrategy):
    """
    Implementation of the Social Infotaxis Exploration Strategy.
    This algorithm computes the next destination based on entropy reduction, information gain,
    and Bayesian belief updates to locate a source efficiently.
    """

    def __init__(self, tau: float = 0.1, epsilon: float = 0.1, **kwargs):
        """
        Parameters:
        ----------
        tau : float
            Temperature parameter for softmax action selection.
        epsilon : float
            Exploration probability for choosing random actions.
        """
        super().__init__(**kwargs)
        self.tau = tau
        self.epsilon = epsilon

    def choose_destination(
        self,
        current_position=np.empty((0, 2)),
        success_locs=np.empty((0, 2)),
        failure_locs=np.empty((0, 2)),
        other_agent_locs=np.empty((0, 2)),
    ):
        """
        Choose the next destination based on the Social Infotaxis algorithm.

        Parameters:
        ----------
        current_position : tuple
            Current position of the agent (x, y).
        success_locs : np.ndarray
            Array of successful resource locations visited by the agent.
        failure_locs : np.ndarray
            Array of failed resource locations visited by the agent.
        other_agent_locs : np.ndarray
            Array of locations of other agents.

        Returns:
        -------
        np.ndarray
            Selected destination (x, y).
        """
        # Initialize belief as uniform if not already initialized
        if not hasattr(self, "belief"):
            self.belief = np.ones((self.grid_size, self.grid_size)) / (
                self.grid_size**2
            )

        # Update belief with social information (e.g., from other agents)
        self._update_belief_with_social_info(other_agent_locs)

        # Compute current entropy
        current_entropy = self._compute_entropy(self.belief)

        # Compute expected entropy for all possible actions
        action_set = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])  # Example actions
        expected_entropies = np.array(
            [
                self._compute_expected_entropy(current_position, action, self.belief)
                for action in action_set
            ]
        )

        # Compute information gain
        information_gain = current_entropy - expected_entropies

        # Compute softmax probabilities for action selection
        probabilities = np.exp(information_gain / self.tau) / np.sum(
            np.exp(information_gain / self.tau)
        )

        # Choose an action
        if np.random.rand() < self.epsilon:  # Exploration
            chosen_action = action_set[np.random.choice(len(action_set))]
        else:  # Exploitation
            chosen_action = action_set[np.argmax(probabilities)]

        # Compute new position
        new_position = np.clip(current_position + chosen_action, 0, self.grid_size - 1)

        # Update belief state
        self._update_belief(self.belief, new_position)

        return new_position

    def _compute_entropy(self, belief):
        """
        Compute the entropy of the current belief distribution.

        Parameters:
        ----------
        belief : np.ndarray
            Current belief distribution.

        Returns:
        -------
        float
            Entropy of the belief distribution.
        """
        return -np.sum(
            belief * np.log(belief + 1e-9)
        )  # Add a small value to prevent log(0)

    def _compute_expected_entropy(self, current_position, action, belief):
        """
        Compute the expected entropy after taking a given action.

        Parameters:
        ----------
        current_position : tuple
            Current position of the agent (x, y).
        action : np.ndarray
            Action to evaluate (x, y offsets).
        belief : np.ndarray
            Current belief distribution.

        Returns:
        -------
        float
            Expected entropy after taking the action.
        """
        # Predict the next position
        new_position = np.clip(current_position + action, 0, self.grid_size - 1)

        # Approximate the new belief state using Bayesian inference
        new_belief = belief.copy()
        self._update_belief(new_belief, new_position)

        # Compute entropy of the new belief state
        return self._compute_entropy(new_belief)

    def _update_belief(self, belief, new_position):
        """
        Update the belief state using Bayesian inference.

        Parameters:
        ----------
        belief : np.ndarray
            Current belief distribution to update.
        new_position : tuple
            New position of the agent (x, y).
        """
        # Placeholder for observation likelihood (this would be domain-specific)
        observation_likelihood = np.random.uniform(0.1, 1.0, belief.shape)

        # Bayesian update
        belief *= observation_likelihood
        belief /= np.sum(belief)  # Normalize the belief distribution

    def _update_belief_with_social_info(self, other_agent_locs):
        """
        Update the belief state with social information (e.g., other agents' sampling locations).
        """
        if len(other_agent_locs) > 0:
            for loc in other_agent_locs:
                x, y = loc
                self.belief[x, y] += 1  # Boost belief at other agents' locations
            self.belief /= np.sum(self.belief)  # Normalize the belief distribution


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    _rng = np.random.default_rng(42)

    grid_size = 90
    exploration_strategy = KernelBeliefExploration(
        grid_size=grid_size,
        tau=0.5,
        social_length_scale=25.0,
        success_length_scale=10.0,
        failure_length_scale=10.0,
        w_social=0.1,
        w_success=0.2,
        w_failure=0.1,
        w_locality=0.6,
        w_as_attention_shares=True,
        model_type="kde",
        normalize_features=True,
        rng=_rng
    )

    current_position = np.array([[30, 20]], dtype=float)

    def rand_locs(n):
        return _rng.integers(0, grid_size, size=(n, 2), endpoint=False, dtype=np.int64).astype(float)

    other_agent_locs = rand_locs(5)
    success_locs = rand_locs(5)
    failure_locs = rand_locs(5)

    exploration_strategy.softmax_tau = 0.1
    dest = exploration_strategy.choose_destination(
        current_position=current_position,
        success_locs=success_locs,
        failure_locs=failure_locs,
        other_agent_locs=other_agent_locs,
    )

    exploration_strategy.softmax_tau = 1.0
    _ = exploration_strategy.choose_destination(
        current_position=current_position,
        success_locs=success_locs,
        failure_locs=failure_locs,
        other_agent_locs=other_agent_locs,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    extent = (0, grid_size, 0, grid_size)

    panels = [
        (
            "Social feature (others)",
            exploration_strategy.social_feature_kde,
            other_agent_locs,
        ),
        ("Success feature", exploration_strategy.success_feature_kde, success_locs),
        ("Failure feature", exploration_strategy.failure_feature_kde, failure_locs),
        ("Softmax (weighted sum)", exploration_strategy.belief_softmax, None),
    ]

    for n, (ax, (title, field, pts)) in enumerate(zip(axes.flat, panels)):
        if n != 3:
            im = ax.imshow(
                field,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                vmin=-5,
                vmax=5,
                cmap="PuOr"
            )
        else:
            im = ax.imshow(
                field, origin="lower", extent=extent, interpolation="nearest",
                cmap="cividis"
            )
        ax.set_title(title)
        ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)
        ax.set_xlabel("x"); ax.set_ylabel("y")

        if pts is not None and pts.size:
            ax.scatter(
                pts[:, 1], pts[:, 0],
                s=36, facecolors="none",
                edgecolors="#1b9e77",  # "white",
                linewidths=1.2,
                label="obs"
            )

        ax.scatter(current_position[0, 1], current_position[0, 0],
                   s=60, marker="o", facecolors="none", edgecolors="black",
                   linewidths=1.2, label="agent")
        ax.scatter(dest[0], dest[1],
                   s=90, marker="*", facecolors="red", edgecolors="black",
                   linewidths=1.0, label="destination")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.show()
