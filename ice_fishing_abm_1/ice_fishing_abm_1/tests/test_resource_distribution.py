import mesa
import pytest

from ice_fishing_abm_1.ice_fishing_abm_1.resource_distribution import ResourceDistribution, \
    generate_resource_map_with_make_blobs


@pytest.fixture
def resource():
    # create a mesa model
    m = mesa.Model()
    m.grid = mesa.space.MultiGrid(100, 100, torus=False)

    r = ResourceDistribution(m)
    return r


@pytest.mark.parametrize("cluster_std", [0.1, 0.5, 1.0])
def test_generate_resource_map(cluster_std):
    """
    Test the generate_resource_map function
    """
    # test the function with default parameters
    resource_map = generate_resource_map_with_make_blobs(100, 100, cluster_std=cluster_std, n_samples=2000,
                                                         normalize=False)

    assert resource_map.shape == (100, 100), "The shape of the resource map should be (100, 100)"
    assert resource_map.sum() == 2000, "The number of fish should be equal to n_samples"
    assert resource_map.min() >= 0, "The minimum value of the resource map should be greater than 0"

    # test the function with normalize=True
    resource_map = generate_resource_map_with_make_blobs(100, 100, cluster_std=cluster_std, n_samples=2000,
                                                         normalize=True)

    assert resource_map.shape == (100, 100), "The shape of the resource map should be (100, 100)"
    assert resource_map.max() <= 1, "The maximum value of the resource map should be less than or equal to 1"
    assert resource_map.min() >= 0, "The minimum value of the resource map should be greater than 0"


def test_make_cluster_centers(resource):
    centers = resource.make_cluster_centers(n_clusters=5, cluster_radius=5)
    assert len(centers) == 5, "There should be 5 cluster centers"
