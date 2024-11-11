import unittest
import numpy as np
from abm.exploration_strategy import ExplorationStrategy, RandomWalkerExplorationStrategy

class TestExplorationStrategy(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        self.strategy = ExplorationStrategy(grid_size=10)
    
    def test_choose_destination_trivial(self):
        # Basic test to ensure destination is within grid bounds
        destination = self.strategy.choose_destination(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)))
        x, y = destination
        self.assertTrue(0 <= x < self.strategy.grid_size)
        self.assertTrue(0 <= y < self.strategy.grid_size)
    
    def test_choose_destination_probability_distribution(self):
        # Check if belief_softmax is normalized and has the expected shape
        self.assertAlmostEqual(np.sum(self.strategy.belief_softmax), 1.0, places=5)
        self.assertEqual(self.strategy.belief_softmax.shape, (self.strategy.grid_size, self.strategy.grid_size))

    def test_invalid_inputs(self):
        # Verify that invalid input raises an assertion error
        with self.assertRaises(AssertionError):
            self.strategy.choose_destination(np.array([1, 2]), np.empty((0, 2)), np.empty((0, 2)))

        with self.assertRaises(AssertionError):
            self.strategy.choose_destination(np.empty((0, 2)), np.array([[1, 2, 3]]), np.empty((0, 2)))


class TestRandomWalkerExplorationStrategy(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        self.random_walker = RandomWalkerExplorationStrategy(grid_size=10, mu=1.5, dmin=1.0, L=5.0, alpha=0.1)
    
    def test_levy_flight_trivial(self):
        # Ensure destination is within grid bounds
        destination = self.random_walker._levy_flight()
        x, y = destination
        self.assertTrue(0 <= x < self.random_walker.grid_size)
        self.assertTrue(0 <= y < self.random_walker.grid_size)
    
    def test_levy_flight_specific_values(self):
        # Use a specific random seed to check for expected output
        np.random.seed(1)  # Seed for reproducibility
        destination = self.random_walker._levy_flight()
        #print(destination)
        expected_destination = np.array([4, 3])  # Expected values with this seed and parameters
        np.testing.assert_array_equal(destination, expected_destination)

    def test_adjust_for_social_cue(self):
        # Test with a social cue close to the walker to see if it adjusts
        social_locs = np.array([[5, 5], [1, 1]])
        np.random.seed(1)  # For consistency in randomness
        adjusted_destination = self.random_walker._adjust_for_social_cue(np.array([4, 4]), social_locs)
        
        # Check if it switched to the nearest social cue
        self.assertTrue(np.array_equal(adjusted_destination, [5, 5]) or np.array_equal(adjusted_destination, [4, 4]))

    def test_destination_with_social_cue(self):
        # Test if the walker selects a destination considering social cues
        social_locs = np.array([[3, 3], [7, 7]])
        destination = self.random_walker.choose_destination(social_locs, np.empty((0, 2)), np.empty((0, 2)), social_cue=True)
        x, y = destination
        self.assertTrue(0 <= x < self.random_walker.grid_size)
        self.assertTrue(0 <= y < self.random_walker.grid_size)

    def test_get_new_position(self):
        # Check new position calculation with wrapping around the grid
        dx, dy = 3, 3
        new_position = self.random_walker._get_new_position(dx, dy)
        expected_position = np.array([8, 8])  # Assuming center start in a 10x10 grid
        np.testing.assert_array_equal(new_position, expected_position)


if __name__ == '__main__':
    unittest.main()
