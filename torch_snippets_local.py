import PIL
import numpy as np
import tqdm
def Tqdm(x, total=None, desc=None):
    total = len(x) if total is None else total
    return tqdm.tqdm(x, total=total, desc=desc)

def flatten(lists):
    return [y for x in lists for y in x]

import os
import glob
def Glob(x, extns=None):
    files = glob.glob(x + "/*") if "*" not in x else glob.glob(x)
    if extns:
        if isinstance(extns, str):
            extns = extns.split(",")
        files = [f for f in files if any([f.endswith(ext) for ext in extns])]
    return files

def fname(fpath):
    return fpath.split("/")[-1]

def stem(fpath):
    name = fpath
    i = name.rfind('.')
    if 0 < i < len(name) - 1:
        return name[:i]
    else:
        return name

import random
def choose(imgs):
    return imgs[random.randint(0, len(imgs)-1)]


import cv2
def readImage2RGB(fpath):
    '''read an image from a file to RGB format'''
    img = cv2.imread(str(fpath))
    # convert BGR to RGB
    if img is None:
        raise ValueError(f"Image not found: {fpath}")
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]
    #img = img[..., ::-1]  # BGR to RGB
    return img

def readImage2CV2(fpath):
    '''read an image from a file to RGB format'''
    img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
    return img

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

def simple_show(
    img=None,
    ax=None,
    title=None,
    sz=None,
    texts=None,
    cmap="gray",
    grid: bool = False,
    save_path: str = None,
    text_sz: int = None,
    font_path=None,
    **kwargs,
):
    "show an image"
    from IPython.display import display
    plt.rcParams["axes.edgecolor"] = "black"
    globals().update(locals())

    try:
        import torch
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy().copy()
    except ModuleNotFoundError:
        pass
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)

    if not isinstance(img, np.ndarray):
        display(img)
        return

    if len(img.shape) == 3 and len(img) == 3:
        # this is likely a torch tensor
        img = img.transpose(1, 2, 0)
    img = np.copy(img)
    if img.max() == 255:
        img = img.astype(np.uint8)
    h, w = img.shape[:2]
    if sz is None:
        if w < 50:
            sz = 1
        elif w < 150:
            sz = 2
        elif w < 300:
            sz = 5
        elif w < 600:
            sz = 10
        else:
            sz = 20
    if isinstance(sz, int):
        sz = (sz, sz)
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", sz))
        _show = True
    else:
        _show = False
    text_sz = text_sz if text_sz else (max(sz) * 3 // 5)
    
    if title:
        ax.set_title(title, fontdict=kwargs.pop("fontdict", None))
    ax.imshow(img, cmap=cmap, **kwargs)

    if grid:
        ax.grid()
    else:
        ax.set_axis_off()

    if save_path:
        fig.savefig(save_path)
        return
    if _show:
        plt.show()

import torch
def inspect_shape(*arrays, **kwargs):
    """
    shows shape, min, max and mean of an array/list/dict of oreys
    Usage:
    inspect(arr1, arr2, arr3, [arr4,arr5,arr6], arr7, [arr8, arr9],...)
    where every `arr` is  assume to have a .shape, .min, .max and .mean methods
    """
    depth = kwargs.pop("depth", 0)
    names = kwargs.pop("names", None)
    for ix, arr in enumerate(arrays):
        name = "\t" * depth
        name = (
            name + f"{names[ix].upper().strip()}:\n" + name
            if names is not None
            else name
        )
        name = name
        typ = type(arr).__name__
        if hasattr(arr, "shape"):
            sh, m, M, dtype = arr.shape, arr.min(), arr.max(), arr.dtype
            try:
                me = arr.mean()
            except:
                me = arr.float().mean()
            info = f"{name}{typ}\tShape: {sh}\tMin: {m:.3f}\tMax: {M:.3f}\tMean: {me:.3f}\tdtype: {dtype}"
            if hasattr(arr, "device"):
                info += f" @ {arr.device}"
            print(info)
        else:
            name = name + f"{typ}:\n"
            
def subplots(ims, nc=5, figsize=(5, 5), silent=True, **kwargs):
    plt.rcParams["axes.edgecolor"] = "black"
    if len(ims) == 0:
        return
    titles = kwargs.pop("titles", [None] * len(ims))
    if isinstance(titles, str):
        if titles == "ixs":
            titles = [str(i) for i in range(len(ims))]
        else:
            titles = titles.split(",")
    if len(ims) <= 5 and nc == 5:
        nc = len(ims)
    nr = (len(ims) // nc) if len(ims) % nc == 0 else (1 + len(ims) // nc)
    if not silent:
        print(f"plotting {len(ims)} images in a grid of {nr}x{nc} @ {figsize}")
    figsize = kwargs.pop("sz", figsize)
    figsize = (figsize, figsize) if isinstance(figsize, int) else figsize
    fig, axes = plt.subplots(nr, nc, figsize=figsize)
    return_axes = kwargs.pop("return_axes", False)
    axes = axes.flat
    fig.suptitle(kwargs.pop("suptitle", ""))

    if "text_col" in kwargs:
        text_cols = [kwargs.pop("text_col")] * len(ims)
    else:
        text_cols = kwargs.pop("text_cols", [None] * len(ims))
    titles = titles.split(",") if isinstance(titles, str) else titles
    for ix, (im, ax) in enumerate(zip(ims, axes)):
        simple_show(
            im,
            ax=ax,
            title=titles[ix],
            **kwargs,
        )
    blank = np.eye(100) + np.eye(100)[::-1]
    for ax in axes:
        simple_show(blank, ax=ax)
    plt.tight_layout()
    if return_axes:
        return axes
    plt.show()

def line(string="", lw=66, upper=True, pad="\N{Box Drawings Double Horizontal}"):
    i = string.center(lw, pad)
    if upper:
        i = i.upper()
    print(i)
