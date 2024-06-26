import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier


def generate_belief_matrix(grid_size: int, margin_size: int, X: np.ndarray, y: np.ndarray,
                           gpc: GaussianProcessClassifier, class_index: int = 1):
    gpc.fit(X, y)
    x = np.meshgrid(range(margin_size // 2, grid_size + margin_size // 2),
                    range(margin_size // 2, grid_size + margin_size // 2))
    s_prob = gpc.predict_proba(np.array(x).reshape(2, -1).T)
    return s_prob[:, class_index].reshape(grid_size, grid_size)


def construct_dataset_info(grid_size: int, margin_size: int, locs: np.ndarray, step_size: int = 20):
    # add margin to agent locations
    locs = locs + margin_size // 2

    # meshgrid size
    mesh_size = (grid_size + margin_size) // step_size
    mesh = np.array(np.meshgrid(range(mesh_size), range(mesh_size))) * step_size
    mesh += margin_size // 2

    # mesh to long format
    mesh = mesh.reshape(2, -1).T

    # remove mesh point if it is too close to the agent
    dist = np.linalg.norm(locs[:, None] - mesh, axis=2)
    mask = np.min(dist, axis=0) > step_size
    mesh = mesh[mask]

    # combine agent locations and mesh
    X = np.vstack([locs, mesh])
    y = np.hstack([np.ones(locs.shape[0]), np.zeros(mesh.shape[0])])

    return X, y
