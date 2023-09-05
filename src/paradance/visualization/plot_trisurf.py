from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt


class TrisurfPloter:
    """TrisurfPloter class for plotting trisurf."""

    def __init__(
        self,
        surf_data_group: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        surf_labels: List[str],
        xyz_labels: List[str],
        cmap_colors: List[str] = ["Blues", "Greens", "Oranges"],
        alphas: List[float] = [0.5, 0.6, 0.7],
        legend_loc: str = "lower center",
        view_init: Tuple[float, float] = (10, 35),
        save_fig: bool = False,
        file_tag: Optional[float] = None,
        file_type: str = "pdf",
    ) -> None:
        """Initialize TrisurfPloter.

        :param surf_data_group: surf data group
        :param surf_labels: surf labels
        :param cmap_colors: cmap colors
        :param alphas: alphas
        :param legend_loc: legend location
        :param view_init: view init
        :param save_fig: whether to save the figure
        :param file_tag: file tag
        :param file_type: file type
        """
        self.surf_data_group = surf_data_group
        self.surf_labels = surf_labels
        self.xyz_labels = xyz_labels
        self.cmap_colors = cmap_colors
        self.alphas = alphas
        self.legend_loc = legend_loc
        self.view_init = view_init
        self.save_fig = save_fig
        self.file_tag = file_tag
        self.file_type = file_type
        self.surf_num = len(surf_data_group)
        self.plot_trisurfs()

    def init_figure(self, grid_color: str = "white") -> None:
        """Initialize figure."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.grid(color=grid_color)

    def get_color_patches(self) -> List[mpatches.Patch]:
        """Get color patches for legend.

        :return: color patches
        """
        handles = []
        for surf, surf_label in zip(self.surfs, self.surf_labels):
            handles.append(
                mpatches.Patch(color=surf.get_facecolor()[-1], label=surf_label)
            )
        return handles

    def set_xyz_labels(self) -> None:
        """Set xyz labels.""" ""
        self.ax.set_xlabel(self.xyz_labels[0])
        self.ax.set_ylabel(self.xyz_labels[1])
        self.ax.set_zlabel(self.xyz_labels[2])

    def plot_trisurfs(self) -> None:
        """Plot trisurfs."""
        self.init_figure()
        self.surfs = []
        for surf_data, cmap_color, alpha in zip(
            self.surf_data_group, self.cmap_colors, self.alphas
        ):
            self.surfs.append(
                self.ax.plot_trisurf(
                    surf_data[0].ravel(),
                    surf_data[1].ravel(),
                    surf_data[2].ravel(),
                    cmap=cmap_color,
                    edgecolor="none",
                    antialiased=True,
                    alpha=alpha,
                )
            )
        handles = self.get_color_patches()
        self.set_xyz_labels()
        self.ax.legend(
            handles=handles, loc=self.legend_loc, frameon=False, ncol=self.surf_num
        )
        elev, azim = self.view_init
        self.ax.view_init(elev, azim)
        if self.save_fig:
            self.fig.savefig(f"trisurf_{self.file_tag}.{self.file_type}")
        plt.show()
