import numpy as np

class ExplorationStrategy:
    def __init__(self, grid_size: int = 100, ucb_beta=0.2, tau=0.01):
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
        Select destination randomly using a generic softmax approach
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


class RandomWalkerExplorationStrategy(ExplorationStrategy):
    def __init__(self, social_length_scale: float = 12, success_length_scale: float = 5,
                 failure_length_scale: float = 5, w_social: float = 0.4, w_success: float = 0.3, w_failure: float = 0.3,
                 motion_type="Brownian", sigma: float = 1.0, mu: float = 1.5, dmin: float = 1.0, L: float = 10.0,
                 C: float = 1.0, alpha: float = 0.1, random_state: int = 0, **kwargs):
        super().__init__(**kwargs)

        # For compatibility with GPExplorationStrategy inputs (not actually used in this strategy)
        self.social_length_scale = social_length_scale
        self.success_length_scale = success_length_scale
        self.failure_length_scale = failure_length_scale
        self.w_social = w_social
        self.w_success = w_success
        self.w_failure = w_failure

        # Random walker-specific parameters
        self.motion_type = motion_type  # 'Brownian' or 'Levy'
        self.sigma = sigma  # Brownian step size
        self.mu = mu  # Levy exponent
        self.dmin = dmin
        self.L = L
        self.C = C
        self.alpha = alpha
        self.random_state = random_state

    def choose_destination(self, success_locs, failure_locs, other_agent_locs):
        """
        Choose destination based on motion type (Brownian or Levy)
        """
        # Ensure input data is in correct format
        self._check_input(success_locs)
        self._check_input(failure_locs)
        self._check_input(other_agent_locs)

        # Random walk strategy based on the motion type
        if self.motion_type == "Brownian":
            self.destination = self._brownian_motion()
        elif self.motion_type == "Levy":
            self.destination = self._levy_flight()

        # Ensure destination is integer after computation
        self.destination = np.round(self.destination).astype(int) 

        return self.destination

    def _brownian_motion(self):
        """
        Implement Brownian motion, sampling distance and angle
        """
        # Sample the non-negative distance from a normal distribution
        distance = np.abs(np.random.normal(0, self.sigma))
        # Sample the angle uniformly from [0, 2π]
        angle = np.random.uniform(0, 2 * np.pi)
        # Convert polar coordinates (distance, angle) to Cartesian displacement
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)

        # Find new destination
        destination = self._get_new_position(dx, dy)
        return destination

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
        new_position = np.clip(current_pos + np.array([dx, dy]), 0, self.grid_size - 1)
        return new_position

    def _get_new_position(self, dx, dy):
        """
        Compute the new position from a displacement (dx, dy)
        """
        # Current position (we assume the walker starts from the center of the grid)
        current_position = np.array([self.grid_size // 2, self.grid_size // 2])

        # Calculate new position with periodic boundary conditions (wrap around the grid)
        new_x = (current_position[0] + dx) % self.grid_size
        new_y = (current_position[1] + dy) % self.grid_size

        # Ensure new_x and new_y are integers by rounding
        new_x = int(np.round(new_x))
        new_y = int(np.round(new_y))

        return np.array([new_x, new_y])