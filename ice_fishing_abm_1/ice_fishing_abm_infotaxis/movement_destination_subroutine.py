import numpy as np

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