
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.gridspec as gridspec
import torchvision.transforms.functional as F
from arch_like import draw_arch, cropfun, cjfun


stylef = "../plot/berenslab.mplstyle"
transform=transforms.Compose([
    transforms.ToTensor()
])

img_augmentations = {
    "H-Flip": F.hflip, 
    "Crop": lambda x: cropfun(x),
    "Jitter": cjfun,
    "Grayscale": F.to_grayscale,   
}

our_augmentations = {
    "V-Flip": F.vflip,
    "90° Rot.": lambda x: F.rotate(x, 90),
    "Rand Rot.": lambda x: F.rotate(x, 35, fill=[218.8325, 218.8325, 218.8325])
}
image = Image.open('../figures/images/000200.tiff').resize((28,28))

img = Image.fromarray(np.array(image))
img2 = transform(img)

grid_rows, grid_cols = 2, 7
fig, axes = plt.subplots(figsize=(4.8, 1.5),constrained_layout=True)
gs = gridspec.GridSpec(grid_rows, grid_cols, figure=fig)
ax_a = fig.add_subplot(gs[:, :3])
with plt.style.context(stylef):
    draw_arch(ax_a, img2)

for i, (title, tr) in enumerate(img_augmentations.items()):
    ax = fig.add_subplot(gs[0, 3 + i])
    ax.set_axis_off()
    ax.imshow(tr(img), cmap="gray")
    title_y_position = -0.15
    ax.text(0.5, title_y_position, title, ha='center', va='top', transform=ax.transAxes, fontsize=7)

for i, (title, tr) in enumerate(our_augmentations.items()):
    ax = fig.add_subplot(gs[1, 3 + i])
    ax.set_axis_off()
    ax.imshow(tr(img), cmap="gray")
    title_y_position = -0.15
    ax.text(0.5, title_y_position, title, ha='center', va='top', transform=ax.transAxes, fontsize=7)


axes.set_axis_off()
fig.text(0.005, 1., 'a', ha='left', va='top', fontsize=7, fontweight='bold')  
fig.text(0.47, 1., 'b', ha='left', va='top', fontsize=7, fontweight='bold')  
fig.text(0.47, 0.5, 'c', ha='left', va='top', fontsize=7, fontweight='bold')  

plt.savefig('../figures/arch.png',dpi=100)
plt.savefig('../figures/arch.pdf',dpi=100)

