
import random
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import matplotlib.pyplot as plt
from architecture import draw_arch
from torchvision import transforms
import torchvision.transforms.functional as F
random.seed(10)

def cjfun(t):
    t = F.adjust_brightness(t, 0.5)
    t = F.adjust_contrast(t, 0.4)
    t = F.adjust_saturation(t, 0.1)
    t = F.adjust_hue(t, 0.2)
    return t

def cropfun(t):
    scale = (0.08, 1.0)
    ratio = (0.75, 1.3333333333333333)

    image_height,image_width=28,28
    random_scale = random.uniform(scale[0], scale[1])

    random_ratio = random.uniform(ratio[0], ratio[1])

    height = int(random_scale * random_ratio * image_height)
    width = int(random_scale * image_width)

    top = random.randint(0, image_height - height)
    left = random.randint(0, image_width - width)
    return F.resized_crop(t, top=top, left=left, height=height, width=width, size=[28, 28])


transform=transforms.Compose([
    transforms.ToTensor()
])

img_augmentations = {
    "H-Flip": F.hflip, 
    "Crop": cropfun,
    "Jitter": cjfun,
    "Grayscale": F.to_grayscale,   
}

our_augmentations = {
    "V-Flip": F.vflip,
    "90° Rot.": lambda x: F.rotate(x, 90),
    "Rand Rot.": lambda x: F.rotate(x, 35, fill=[218.8325, 218.8325, 218.8325])
}

original_image=Image.open('figures/images/000200.tiff').resize((32,32))

grid_rows, grid_cols = 2, 7
all_axes = [] 

fig, axes = plt.subplots(figsize=(6, 2),constrained_layout=True)

ax_a = plt.subplot2grid((grid_rows, grid_cols), (0, 0), rowspan=grid_rows, colspan=3)
original_image2=transform(original_image)

stylef='berenslab.mplstyle'
with plt.style.context(stylef):
    draw_arch(ax_a, original_image2)
all_axes.append(ax_a)

img_dict = dict(ha='center', va='top', fontsize=7)
for i, (title, tr) in enumerate(img_augmentations.items()):
    ax = plt.subplot2grid((grid_rows, grid_cols), (0, 3 + i)) 
    ax.set_axis_off()
    ax.imshow(tr(original_image), cmap="gray")
    ax.text(0.5, -0.15, title, transform=ax.transAxes,fontdict=img_dict )
    all_axes.append(ax)

for i, (title, tr) in enumerate(our_augmentations.items()):
    ax = plt.subplot2grid((grid_rows, grid_cols), (1, 3 + i))  
    ax.set_axis_off()
    ax.imshow(tr(original_image), cmap="gray")
    ax.text(0.5, -0.15, title, transform=ax.transAxes,fontdict=img_dict)
    all_axes.append(ax)

axes.set_axis_off()
text_dict= dict(ha='left', va='top', fontsize=7, fontweight='bold')
fig.text(0.005, 1., 'a',fontdict=text_dict)  
fig.text(0.47, 1., 'b',fontdict=text_dict)  
fig.text(0.47, 0.5, 'c',fontdict=text_dict)  

plt.savefig('figures/arch-augmentation.pdf',dpi=300)
plt.savefig('figures/arch-augmentation.png',dpi=100)