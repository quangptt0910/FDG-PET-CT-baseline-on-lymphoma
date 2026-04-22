import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def save_figure(fig, path: Path, dpi: int = 150) -> None:
    """
    Save a matplotlib figure to disk.

    Args:
        fig: Matplotlib figure object
        path: Path where to save the figure
        dpi: Resolution in dots per inch (default: 150)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_overlay_panel(
    ax, pet_slice: np.ndarray, mask_slice: np.ndarray, color: str, alpha: float = 0.4
) -> None:
    """
    Create an overlay panel showing PET image with mask overlay.

    Args:
        ax: Matplotlib axes object
        pet_slice: 2D PET image slice
        mask_slice: 2D binary mask slice
        color: Color for mask overlay (e.g., 'red', 'green')
        alpha: Transparency of mask overlay (default: 0.4)
    """
    # Normalize PET slice for display
    pet_norm = (pet_slice - np.min(pet_slice)) / (
        np.max(pet_slice) - np.min(pet_slice) + 1e-8
    )

    # Show PET image
    ax.imshow(pet_norm.T, cmap="hot", origin="lower")

    # Create colored mask overlay
    if np.any(mask_slice):
        # Create RGBA array for overlay
        overlay = np.zeros((*mask_slice.shape, 4))
        if color == "red":
            overlay[..., 0] = 1.0  # Red channel
        elif color == "green":
            overlay[..., 1] = 1.0  # Green channel
        overlay[..., 3] = mask_slice * alpha  # Alpha channel based on mask

        # Show overlay
        ax.imshow(overlay, origin="lower")

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
