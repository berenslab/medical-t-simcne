import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from annotate_pathmnist import annotate_path
rng = np.random.default_rng(0x3FF21)
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import medmnist,torch
from PIL import Image,ImageOps
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

class GetFirst(object):
    def __init__(self, ar):
        super().__init__()
        self.ar = ar

    def __getitem__(self, i):
        return self.ar[i][0]
    
def add_border(input_image, border_size, border_color):
        img_with_border = ImageOps.expand(input_image, border=border_size, fill=border_color)
        return img_with_border
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
stylef = "berenslab.mplstyle"

npz2 = np.load(f'{root}/PathMNIST2.npz', allow_pickle=True)
Y2 = npz2['Med_Default_with_Rot_0'].item()['data']
labels_ = np.load(f"{root}/label_PathMNIST.npy")
train = medmnist.PathMNIST(
    "train", root=root, download=False
)
test = medmnist.PathMNIST(
    "test", root=root, download=False
)
val = medmnist.PathMNIST(
    "val", root=root, download=False
)
dataset = torch.utils.data.ConcatDataset([train, test, val])



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
    print(images.shape)
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
                # print(f"{i}, {j}, {majority_class}, {in_grid.sum(): 6d}")

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

    np.save(f"{root}/pixels.npy", pixels)
    print("pixels saved")


with plt.style.context("berenslab.mplstyle"):
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(4.8, 1.7),
        layout="compressed",
        constrained_layout=True,
        # subplot_kw=dict(bottom=0, left=0, top=1, right=1),
    )

    axs[1].scatter(*Y.T, c=labelcolors[labels], alpha=0.5, rasterized=True)
    axs[2].imshow(pixels, origin="lower")
    axs[0].scatter(
        Y2[:, 0],
        Y2[:, 1],
        c=labels_,
        alpha=0.5,
        rasterized=True,
    )
    

    annotate_path(axs[0], Y2, dataset)
    
    handles = [
        mpl.lines.Line2D(
            [], [], lw=0, marker="o", markersize=3, label=name, color=c
        )
        for name, c in zip(labelnames, labelcolors)
    ]

    # fig.legend(handles=markers, 
    #                         ncol=1, loc="upper left", 
    #                         fontsize=7,
    # bbox_to_anchor=(0.73, 0.75),frameon=False,
    #                         handletextpad=0.1, columnspacing=-0.1,
    #  borderaxespad=0)
    axs[2].legend(
        handles=handles,
        fontsize=4,
        loc="upper left",
        bbox_to_anchor=(-0.25, 0.9),
        handletextpad=0.1,
        frameon=False,
        columnspacing=-0.1
    )
    # fd = dict(ha="left", va="top", fontsize=1, weight="bold")

    # axs[0].text(-0.2, 1, 'a', transform=axs[0].transAxes, **fd)
    # axs[1].text(0, 1, 'b', transform=axs[1].transAxes, **fd)
    # axs[2].text(0, 1, 'c', transform=axs[2].transAxes, **fd)
    # [
    #     ax.text(-0.1, 1, ltr, transform=ax.transAxes, **fd)
    #     for ax, ltr in zip(axs, "abc")
    # ]
    # axs[0].axis("equal")
    # axs[1].axis("equal")
    # axs[2].axis("equal")
    [ax.axis("equal") for ax in axs]
    [ax.set_axis_off() for ax in axs]

    fig.savefig("../figures/ifeoma1.pdf",dpi=300)#../figures/
    fig.savefig("../figures/ifeoma2.png",dpi=300)#../figures/

print("-------------Done-------------------")