import numpy as np
from scipy.special import softmax

class PatchEvaluationSubroutine:
    def __init__(self, threshold, **kwargs):
        # Initialize default parameters
        self.threshold = threshold

    def stay_on_patch(self, time_on_patch, time_since_last_catch):
        """
        Determines whether the agent should stay on the current patch based on the threshold.
        """
        if (time_on_patch > self.threshold) and (time_since_last_catch > self.threshold):
            return False
        return True


class InfotaxisPatchEvaluationSubroutine(PatchEvaluationSubroutine):
    """
    Infotaxis-based patch evaluation strategy for agents exploring a grid environment.
    The agent selects actions based on expected information gain and updates beliefs using Bayesian inference.
    """

    def __init__(self, grid_size: int = 20, threshold: int = 1, tau: float = 1.0, epsilon: float = 0.1, **kwargs):
        super().__init__(threshold)
        self.grid_size = grid_size
        self.tau = tau  # Temperature parameter for softmax action selection
        self.epsilon = epsilon  # Exploration probability (epsilon-greedy)
        
        # Initialize belief state (uniform distribution over the grid)
        self.belief = np.ones((self.grid_size, self.grid_size)) / (self.grid_size * self.grid_size)

    def stay_on_patch(self, time_on_patch, time_since_last_catch):
        """
        Use the default method to decide whether to stay on a patch.
        """
        return super().stay_on_patch(time_on_patch, time_since_last_catch)

    def update_belief(self, xa, success_locs, failure_locs):
        """
        Update the belief distribution using Bayesian inference based on observations.
        Bayesian update for belief:
        p(x)' = Pr(o|xa, x) * p(x) / (sum over x' Pr(o|xa, x') * p(x'))
        """
        for loc in success_locs:
            self._update_belief_loc(loc, increase=True)
        for loc in failure_locs:
            self._update_belief_loc(loc, increase=False)

        # Normalize belief to maintain probability distribution
        self.belief /= np.sum(self.belief)

    def _update_belief_loc(self, loc, increase=True):
        """
        Update the belief at a specific location based on a success or failure.
        """
        x, y = loc
        radius = 3  # Define a radius around the location to update belief

        for i in range(max(0, x - radius), min(self.grid_size, x + radius + 1)):
            for j in range(max(0, y - radius), min(self.grid_size, y + radius + 1)):
                if increase:
                    self.belief[i, j] += 1  # Increase belief near success location
                else:
                    self.belief[i, j] *= 0.5  # Decrease belief near failure location

    def compute_expected_entropy(self, xa, action):
        """
        Compute the expected entropy H(s|a) at a given location (action).
        Entropy measures the uncertainty of the belief at that location.
        """
        x, y = action
        belief_at_action = self.belief[x, y]
        entropy_at_action = -belief_at_action * np.log(belief_at_action + 1e-10)  # Avoid log(0)
        return entropy_at_action

    def compute_information_gain(self, xa, action):
        """
        Compute the information gain G(s, a) = H(s) - H(s|a) for a given action.
        """
        current_entropy = self.compute_entropy()  # Entropy at current state
        expected_entropy = self.compute_expected_entropy(xa, action)  # Entropy after taking action
        information_gain = current_entropy - expected_entropy  # Gain in information
        return information_gain

    def compute_entropy(self):
        """
        Compute the total entropy H(s) over the entire belief distribution.
        """
        entropy = -np.sum(self.belief * np.log(self.belief + 1e-10))  # Avoid log(0)
        return entropy

    def choose_patch(self, xa, success_locs, failure_locs):
        """
        Select the next patch to explore using the infotaxis strategy.

        1. Update the belief based on the current agent location and past successes/failures.
        2. Compute the expected information gain for each action.
        3. Use a softmax function to select the next action, balancing exploration and exploitation.
        """
        # Step 1: Update the belief distribution using Bayesian inference
        self.update_belief(xa, success_locs, failure_locs)

        # Step 2: Compute information gain for each possible action
        actions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        information_gains = [self.compute_information_gain(xa, action) for action in actions]

        # Step 3: Compute softmax probabilities for action selection based on information gain
        softmax_probs = softmax(np.array(information_gains) / self.tau)

        # Step 4: Epsilon-greedy action selection for exploration-exploitation
        if np.random.rand() < self.epsilon:
            # Exploration: select a random action
            selected_action = actions[np.random.choice(len(actions))]
        else:
            # Exploitation: select the action with the maximum softmax probability
            selected_action = actions[np.argmax(softmax_probs)]

        # Return the selected action as the next destination
        return selected_action

    def move_to_new_position(self, xa, selected_action):
        """
        Execute the action by moving the agent to the new position based on the selected action.
        """
        return selected_action
