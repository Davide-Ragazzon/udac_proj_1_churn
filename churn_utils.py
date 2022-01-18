"""
Module with helper functions
"""

import os
import constants as const
import matplotlib.pyplot as plt


def save_into_folder(fig, fig_name, logger, folder=const.IMG_FOLDER):
    logger.info(f"Saving {fig_name}.png")
    file_png = os.path.join(folder, f"{fig_name}.png")
    fig.savefig(file_png)
    # Added this because having the figure always displayed can get very
    # annoying...
    plt.close(fig)
