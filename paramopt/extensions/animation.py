from typing import List

from natsort import natsorted
from PIL import Image


def create_gif(
    fp: str,
    img_fps: List[str],
    duration: float = 1.0,
    loop: int = 0
) -> None:
    """Generate gif video from images.

    Parameters
    ----------
    fp : str
        Name of the output video
    img_fps : list of strs
        List of image paths
    duration : float, optional
        Time interval between frames [s], by default 1.0.
    loop : int, optional
        Number of loop, by default 0 (infinite)
    """
    if len(img_fps) == 0:
        print('No file')
        return
    images = list(map(lambda file: Image.open(file), natsorted(img_fps)))
    images[0].save(
        fp, save_all=True, append_images=images[1:],
        duration=float(duration)*1000, loop=int(loop))
    print('Done')


def select_images() -> List[str]:
    """Select image files using GUI dialog.

    Returns
    -------
    list of str
        List of image paths
    """
    from tkinter import Tk, filedialog
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(True)
    except:
        pass
    Tk().withdraw()
    img_fps = filedialog.askopenfilenames(
        filetypes=[
            ('PNG files', '*.png'), ('JPEG files', '*.jpg')], initialdir='.')
    return img_fps
