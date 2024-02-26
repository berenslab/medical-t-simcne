import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

rng = np.random.default_rng(0x3FF21)

labelcolors = np.array(
    [mpl.colors.to_hex("tab:orange"),
      mpl.colors.to_hex("tab:blue")]
)
labelnames = ["no metastasis", "metastasis"]

root='../numpy_files'
npz = np.load(f"{root}/camelyon.npz")
labels = npz["lbls"]
n_labels = len(labelnames)
Y = npz["embeddings"]

# number of gridcells in one dimension
nx_grid = 10
ny_grid = 10

xs = np.linspace(Y.min(), Y.max(), nx_grid + 1)
ys = np.linspace(Y.min(), Y.max(), ny_grid + 1)

# this takes a while
try:
    pixels=np.load(f"{root}/pixels.npy")
except FileNotFoundError:
    images = npz["images"]
    px_border = 3
    im_border = ImageOps.expand(Image.fromarray(images[0]), border=px_border)
    im_white = np.full_like(im_border, 255)

    imgs_arranged = np.empty(
        (nx_grid, ny_grid, *im_white.shape), dtype=im_white.dtype
    )

    for i in range(nx_grid):
        for j in range(ny_grid):
            Y0 = Y[:, 0]
            Y1 = Y[:, 1]
            in_grid = (
                (xs[i] <= Y0)
                & (Y0 < xs[i + 1])
                & (ys[j] <= Y1)
                & (Y1 < ys[j + 1])
            )

            if in_grid.sum() >= 100:
                majority_class = np.argmax(
                    [((labels == i) & in_grid).sum() for i in range(n_labels)]
                )
                print(f"{i}, {j}, {majority_class}, {in_grid.sum(): 6d}")

                im = Image.fromarray(
                    rng.choice(images[in_grid & (labels == majority_class)])
                )
                im_border = ImageOps.expand(
                    im, border=px_border, fill=labelcolors[majority_class]
                )

                imarr = np.array(im_border)
            else:
                imarr = im_white

            imgs_arranged[j, i] = imarr

    # Here we move the axis in the array around such that the reshaping
    # preserves the order of the pixels that we want to plot.
    pixels = np.transpose(imgs_arranged, (0, 2, 1, 3, 4)).reshape(
        nx_grid * im_white.shape[0], ny_grid * im_white.shape[1], 3
    )

    np.save("pixels.npy", pixels)
    print("pixels saved")


with plt.style.context("berenslab.mplstyle"):
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(6, 3),
        layout="compressed",
        # subplot_kw=dict(bottom=0, left=0, top=1, right=1),
    )

    axs[0].scatter(*Y.T, c=labelcolors[labels], alpha=0.5, rasterized=True)
    axs[1].imshow(pixels, origin="lower")

    handles = [
        mpl.lines.Line2D(
            [], [], lw=0, marker="o", markersize=5, label=name, color=c
        )
        for name, c in zip(labelnames, labelcolors)
    ]
    axs[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, 0.85),
        handletextpad=0.25,
        frameon=False,
    )

    fd = dict(ha="left", va="top", fontsize="x-large", weight="bold")
    [
        ax.text(0, 1, ltr, transform=ax.transAxes, **fd)
        for ax, ltr in zip(axs, "ab")
    ]
    axs[0].axis("equal")
    [ax.set_axis_off() for ax in axs]

    fig.savefig("../figures/camelyon_annotation.pdf")
    fig.savefig("../figures/camelyon_annotation.png")