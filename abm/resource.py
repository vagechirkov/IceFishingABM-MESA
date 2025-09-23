import mesa
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import imageio.v2 as imageio

class Resource(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        radius: int = 5,
        max_value: int = 100,
        current_value: int = 50, 
        keep_overall_abundance: bool = True,
        neighborhood_radius: int = 20,
    ):
        super().__init__(unique_id, model)
        self.radius: int = radius
        self.model = model
        self.max_value: int = max_value
        self.current_value: int = current_value
        self.keep_overall_abundance: bool = keep_overall_abundance
        self.neighborhood_radius: int = neighborhood_radius
        self.is_resource = True
        self.const_resource: bool = False 
        self.const_catch_probability: bool = True
        self.is_depleted = False

    @property
    def catch_probability(self):
        # Linear relation for catch probability
        if self.const_catch_probability:
            return 1
        else:
            return self.current_value / self.max_value

    def catch(self):
        if (self.model.random.random() < self.catch_probability
            and self.current_value > 0):
            
            if self.const_resource:  # Constant resource doesn't deplete
                return True
            
            self.current_value -= 1
            self.model.total_consumed_resource += 1 
            
            # Check if resource is depleted
            if self.current_value <= 0:
                if self.keep_overall_abundance:
                    # Instead of adding to neighbor, relocate this resource
                    self.relocate_resource()
                self.is_depleted = True
            
            return True
        else:
            return False
        
    def find_new_location(self):
        """Find a new location that doesn't overlap with existing resources."""
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Generate random coordinates
            x = self.model.random.randint(self.radius, self.model.grid.width - self.radius)
            y = self.model.random.randint(self.radius, self.model.grid.height - self.radius)
            
            # Get all nearby cells within twice the resource radius
            nearby_cells = self.model.grid.get_neighborhood(
                (x, y), 
                moore=True, 
                include_center=True, 
                radius=self.radius * 2
            )
            
            # Check if any nearby cells contain resources
            has_nearby_resource = False
            for cell in nearby_cells:
                cell_contents = self.model.grid.get_cell_list_contents(cell)
                if any(isinstance(agent, Resource) for agent in cell_contents):
                    has_nearby_resource = True
                    break
            
            if not has_nearby_resource:
                return (x, y)
            
            attempts += 1
        
        return None  # Return None if no valid location found

    def relocate_resource(self):
        """Relocate the resource to a new valid location."""
        new_pos = self.find_new_location()
        if new_pos:
            self.model.grid.move_agent(self, new_pos)
            self.current_value = self.max_value  # Reset resource value
            self.is_depleted = False
            return True
        return False

    def resource_map(self) -> np.ndarray:
        # Generate resource map based on resource radius
        _resource_map = np.zeros((self.model.grid.width, self.model.grid.height))
        iter_neighborhood = self.model.grid.iter_neighborhood(
            self.pos, moore=False, include_center=True, radius=self.radius
        )
        for n in iter_neighborhood:
            _resource_map[n] = 1
        return _resource_map.astype(float) * self.catch_probability


def make_resource_centers(
    model, n_clusters: int = 1, cluster_radius: int = 5
) -> tuple[tuple[int, int], ...]:
    """make sure that the cluster centers are not too close together"""
    centers = []

    while len(centers) < n_clusters:
        x = model.random.randint(cluster_radius, model.grid.width - cluster_radius)
        y = model.random.randint(cluster_radius, model.grid.height - cluster_radius)

        if len(centers) == 0:
            centers.append((x, y))
            continue

        # Check if the new center is not too close to any of the existing centers
        if np.all(
            np.linalg.norm(np.array(centers) - np.array((x, y)), axis=-1)
            > cluster_radius * 2
        ):
            centers.append((x, y))
    return tuple(centers)


def spatiotemporal_fish_density(
    rng,
    length_scale_time=15.0,
    length_scale_space=1.0,
    n_x=30,
    n_y=30,
    n_time=50,
    sigma_noise=0.0,
    n_samples=1,
    bias=0.0,  # used for sigmoid; larger = less abundance
    temperature=1.0,  # used for sigmoid; larger = flatter
):
    # grids
    xs = np.linspace(0, n_x, n_x)
    ys = np.linspace(0, n_y, n_y)
    ts = np.linspace(0, n_time, n_time)

    # kernels
    def rbf(d, ell):
        return np.exp(-0.5 * (d / ell) ** 2)

    # pairwise distances
    XY = np.array(np.meshgrid(xs, ys, indexing="ij")).reshape(2, -1).T
    Dxy = cdist(XY, XY, "euclidean")
    Dt = cdist(ts[:, None], ts[:, None], "euclidean")

    # covariance matrices
    K_space = rbf(Dxy, length_scale_space)
    K_time = rbf(Dt, length_scale_time)

    # choleskys
    Ls = np.linalg.cholesky(K_space + 1e-9 * np.eye(K_space.shape[0]))
    Lt = np.linalg.cholesky(K_time + 1e-9 * np.eye(K_time.shape[0]))

    # sample fields
    F = np.zeros((n_samples, n_x, n_y, n_time))
    for i in range(n_samples):
        Z = rng.standard_normal((n_x * n_y, n_time))
        F_i = (Ls @ Z) @ Lt.T
        if sigma_noise > 0:
            F_i += sigma_noise * rng.standard_normal(F_i.size)

        # squash to [0,1]
        F_i = 1.0 / (1.0 + np.exp(-(F_i - bias) / temperature))
        F[i] = F_i.reshape(n_x, n_y, n_time)
    return F, xs, ys, ts


def make_fish_density_gif(
    F, xs, ys, ts, sample_idx=0, filename="fish_density.gif", fps=5
):
    frames = []
    field = F[sample_idx]  # shape (n_x, n_y, n_time)
    vmin, vmax = 0, 1  # field.min(), field.max()  # fix color scale

    for ti, t in enumerate(ts):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(
            field[:, :, ti].T,
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(f"t = {int(t)} [min]")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Fish density")

        # save current frame as image buffer
        fig.canvas.draw()
        h, w = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        frames.append(frame)

        plt.close(fig)

    # write gif
    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved to {filename}")


def bias_for_target_mean(p, sigma=1.0, sigma_n=0.0, temperature=1.0):
    """Approximate bias to achieve mean abundance p"""
    if not (0 < p < 1):
        raise ValueError("p must be in (0,1)")
    s2 = sigma**2 + sigma_n**2
    k2 = 1.0 + (np.pi**2 / 3.0) * (s2 / (temperature**2))
    return -temperature * np.sqrt(k2) * np.log(p / (1.0 - p))


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    for t in [0.5, 2]:
        for b in [0, 1, 2, 4]:
            fish_density, _xs, _ys, _ts = spatiotemporal_fish_density(
                rng,
                length_scale_time=15,
                length_scale_space=6,
                n_x=90,
                n_y=90,
                n_time=120,
                n_samples=1,
                temperature=t,
                bias=b,
            )
            print(f"bias = {b}, abundance={fish_density.mean():.3f}")
            make_fish_density_gif(
                fish_density,
                _xs,
                _ys,
                _ts,
                sample_idx=0,
                filename=f"fish_t_{t:.1f}_b_{int(b)}.gif",
                fps=20,
            )
