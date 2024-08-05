import numpy as np

class Agent:
    def __init__(self, model, other_agent_locs, success_locs, failure_locs, mesh, mesh_indices, ucb_beta, softmax_tau):
        self.model = model
        self.other_agent_locs = other_agent_locs
        self.success_locs = success_locs
        self.failure_locs = failure_locs
        self.mesh = mesh
        self.mesh_indices = mesh_indices
        self.ucb_beta = ucb_beta
        self.softmax_tau = softmax_tau
        self._is_moving = False
        self._destination = None
        
        self.social_feature_m = None
        self.social_feature_std = None
        self.success_feature_m = None
        self.success_feature_std = None
        self.failure_feature_m = None
        self.failure_feature_std = None
        self.belief_m = None
        self.belief_std = None
        self.belief_ucb = None
        self.belief_softmax = None
        
        self.social_gpr = None  # Assuming this will be initialized elsewhere
        self.success_gpr = None  # Assuming this will be initialized elsewhere
        self.failure_gpr = None  # Assuming this will be initialized elsewhere
    pass 


class MovementDestinationSubroutine:
    def __init__(self):
        pass
    
    def calculate_features(self, locations, gpr, grid_size):
        if locations.size == 0:
            feature_m = np.zeros((grid_size, grid_size))
            feature_std = np.zeros((grid_size, grid_size))
        else:
            gpr.fit(locations, np.ones(locations.shape[0]))
            feature_m, feature_std = generate_belief_mean_matrix(grid_size, gpr, return_std=True)
        return feature_m.T, feature_std.T

    def compute_beliefs(self):
        self.belief_m = self.model.w_social * self.social_feature_m + \
                        self.model.w_success * self.success_feature_m - \
                        self.model.w_failure * self.failure_feature_m

        self.belief_std = np.sqrt(
            self.model.w_social ** 2 * self.social_feature_std ** 2 +
            self.model.w_success ** 2 * self.success_feature_std ** 2 +
            self.model.w_failure ** 2 * self.failure_feature_std ** 2)

        self.belief_ucb = self.belief_m + self.ucb_beta * self.belief_std

        self.belief_softmax = np.exp(self.belief_ucb / self.softmax_tau) / np.sum(
            np.exp(self.belief_ucb / self.softmax_tau))

    def choose_destination(self):
        self._destination = self.mesh[np.random.choice(self.mesh_indices, p=self.belief_softmax.reshape(-1)), :]

    def movement_destination(self):
        self.social_feature_m, self.social_feature_std = self.calculate_features(
            self.other_agent_locs, self.social_gpr, self.model.grid_size)

        if not self._is_moving:
            self.success_feature_m, self.success_feature_std = self.calculate_features(
                self.success_locs, self.success_gpr, self.model.grid_size)

            self.failure_feature_m, self.failure_feature_std = self.calculate_features(
                self.failure_locs, self.failure_gpr, self.model.grid_size)

        self.compute_beliefs()
        self.choose_destination()
    def choose_destination(self, mesh, mesh_indices, belief_softmax=None):
        # Select a destination patch randomly for the agent from all the avaible patches
        return mesh[np.random.choice(mesh_indices), :]


class GPMovementDestinationSubroutine(MovementDestinationSubroutine):
    def __init__(self):
        super().__init__()

    def choose_destination(self, patch, agent):
        return patch

