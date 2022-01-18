"""
Module with helper functions
"""

import os
import constants as const
import matplotlib.pyplot as plt


def save_into_image_folder(fig, fig_name, logger):
    logger.info(f"Saving {fig_name}.png")
    file_png = os.path.join(const.IMG_FOLDER, f"{fig_name}.png")
    fig.savefig(file_png)
    # Added this because having the figure always displayed can get very
    # annoying...
    plt.close(fig)
