import io
import math

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


def write_gif(path: str, data: np.ndarray, fps: int = 5) -> None:
    """Similar to https://pypi.org/project/array2gif/ but using a different library as backend. Also reverts
    [-1,1] normalization and type conversion from batch_generator, and converts BGR -> RGB colorscheme.

    :param path: Path to file to write GIF to.
    :param data: 4D numpy array with shape=(frames, height, width, channels).
    :param fps: Animation speed of GIF.
    """
    # Revert normalization and type conversion
    anim_data: np.ndarray = data * 127.5 + 127.5
    anim_data = anim_data.astype("uint8")

    # Change color order from BGR to RGB
    anim_data = anim_data[:, :, :, ::-1]

    frames = []
    for i in range(len(anim_data)):
        frames.append(anim_data[i])

    imageio.mimsave(path, frames, "GIF", fps=fps)


def write_gif_grid(path: str, data: np.ndarray, fps: int = 5, dpi: int = None) -> None:
    """Extension of write_gif; takes an array of n*n 4D numpy arrays that are equal in shape and arranges them as a n*n
    grid of gifs.

    :param path: Path to file to write GIF to.
    :param data: 5D numpy array with shape=(video, frames, height, width, channels).
    :param fps: Animation speed of GIF.
    :param dpi: The resolution in dots per inch.
    """
    # Revert normalization and type conversion
    anim_data: np.ndarray = data * 127.5 + 127.5
    anim_data = anim_data.astype("uint8")

    # Change color order from BGR to RGB
    anim_data = anim_data[:, :, :, :, ::-1]

    # Only squared grids supported
    grid_dim = math.floor(math.sqrt(len(anim_data)))
    if grid_dim ** 2 != len(anim_data):
        raise ValueError("5D array must have a number of elements to the power of two!")

    frames = []
    for i in range(anim_data.shape[1]):
        fig = plt.figure(figsize=(grid_dim, grid_dim))

        for j in range(anim_data.shape[0]):
            plt.subplot(grid_dim, grid_dim, j + 1)
            plt.imshow(anim_data[j][i])
            plt.axis('off')

        img_buf = io.BytesIO()
        plt.savefig(img_buf, dpi=dpi, format='png')
        img_buf.seek(0)
        img_bytes = np.asarray(bytearray(img_buf.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img_rgb = img[:, :, ::-1]
        frames.append(img_rgb)
        img_buf.close()
        plt.close(fig)

    imageio.mimsave(path, frames, "GIF", fps=fps)
