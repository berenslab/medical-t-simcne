import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as F
import random
torch.manual_seed(100)

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

def get_image_midpoint(t_im1,xybox_cord,zoom_factor):
        pil_img = to_pil_image(t_im1) 
        img_width, img_height = pil_img.size
        half_width = (img_width * zoom_factor) / 2
        image_center_x, image_center_y = xybox_cord
        top_left_y = image_center_y + img_width
        bottom_left_y = image_center_y - img_width
        left_x = image_center_x - half_width
        midpoint_left_x = left_x  
        midpoint_left_y = (top_left_y + bottom_left_y) / 2  
        midpoint_left = (midpoint_left_x, midpoint_left_y)
        return midpoint_left


def draw_arch(ax, orig_im):
    ax.set_axis_off()

    lower_row = 0.125

    aprops = dict(
        arrowstyle="-|>",
        color="xkcd:dark gray",
        linewidth=plt.rcParams["axes.linewidth"],
        relpos=(1, 0.5),
        shrinkB=7,
        
    )
    txtkwargs = dict(
        usetex=True,
        horizontalalignment="center",
        verticalalignment="center",
    )


    zoom = 1
    
    t_im1 = cropfun(orig_im)
    t_im2 = F.vflip(orig_im)
    new_props = aprops.copy()
   
    xybox_cord=(-0.3, 0.5)
    im0 = OffsetImage(to_pil_image(orig_im), zoom=zoom)
    y_mid=get_image_midpoint(t_im1,xybox_cord,zoom)

    abox0 = AnnotationBbox(
        im0,
        xy=(0.001, y_mid[1]),  
        xybox=xybox_cord, 
        xycoords="axes fraction",
        frameon=False,
        arrowprops=new_props,
        
    )
    ax.add_artist(abox0)
    

    se_props=aprops.copy()
    # se_props["mutation_scale"] = 1
    # se_props["shrinkB"] = 1
    abox1_xybox=(0.25, 0.85)

    im1 = OffsetImage(to_pil_image(t_im1), zoom=zoom)
    y_mid1=get_image_midpoint(t_im1,abox1_xybox,zoom)

    abox1 = AnnotationBbox(
        im1,
        xy=(0.5, y_mid1[1]),
        xybox=abox1_xybox,
        xycoords="axes fraction",
        frameon=False,
        arrowprops=se_props, 
    )
    ax.add_artist(abox1)
    

    im2 = OffsetImage(to_pil_image(t_im2), zoom=zoom)
    abox2_xybox=(0.25, 0.12)
    y_mid2=get_image_midpoint(t_im2,abox2_xybox,zoom)
    abox2 = AnnotationBbox(
        im2,
        xy=(0.5, y_mid2[1]),  
        xybox=abox2_xybox, 
        xycoords="axes fraction",
        frameon=False,
        arrowprops=se_props, 
    )
    ax.add_artist(abox2)

    augprops = dict(
        arrowstyle="<|-|>",
        color="xkcd:dark gray",
        linewidth=plt.rcParams["axes.linewidth"],
        connectionstyle="bar,fraction=0.3",
   
    )

    ax.annotate(
    "",
    (0.15, lower_row),  #two-edged arrow
    (0.15, 0.85),      
    xycoords="axes fraction",
    arrowprops=augprops, 
)
    t = txtkwargs.copy()
    t["usetex"] = False
    t["fontsize"] = 5
    t["transform"] = ax.transAxes
    t["rotation_mode"] = "anchor"
    t["horizontalalignment"] = "left"

    text_x_near_abox1 = abox1_xybox[0] - 0.28 #data augmentation top horizontal position
    text_y_near_abox1 = abox1_xybox[1] + 0.015
    text_x_near_abox2 = abox2_xybox[0] - 0.28 #data augmentation bottom horizontal position
    text_y_near_abox2 = abox2_xybox[1] 
    ax.text(
        text_x_near_abox1,
        text_y_near_abox1,
        s="data aug-\nmentation",
        va="bottom",
        **t,
    )
    ax.text(
        text_x_near_abox2,
        text_y_near_abox2,
        s="data aug-\nmentation",
        va="top",
        **t,
    )

    bkwargs = dict(
        edgecolor="xkcd:slate gray",
        facecolor="xkcd:white",
        linewidth=plt.rcParams["axes.linewidth"],
    )
    # resnet polygon
    xy = np.array([[0.285, 0.04], [0.285, 1.], [0.5, 0.7], [0.5, 0.4]])
    xy[:, 0] += 0.2
    xy[:, 1] -= 0.029

   
    resnet = Polygon(xy, closed=True, **bkwargs)
    t = txtkwargs.copy()
    t["usetex"] = False
    ax.text(*xy.mean(axis=0), "ResNet", **t)
    ax.add_artist(resnet)


    ph_props = aprops.copy()
    ph_props["shrinkB"] = 1
    ph_props["shrinkA"] = 5#how much the arrow start point should move from the specified start position specified


    t = txtkwargs.copy()
    t.update(
        color="xkcd:slate gray",
        va="bottom",
        ha="center",
        fontsize="medium",
        usetex=False,
    )
    max_x = np.max(xy[:, 0])
    rightmost_vertices = xy[xy[:, 0] == max_x]

    upper_rightmost_vertex = rightmost_vertices[rightmost_vertices[:, 1] == np.max(rightmost_vertices[:, 1])][0]

    ax.text(
        upper_rightmost_vertex[0]+0.015,
        upper_rightmost_vertex[1]+0.0075,
        "512",
        **t,
    )
   
    # projection head polygon
    rh_props = aprops.copy()
    rh_props["shrinkA"]=0
    rh_props["shrinkB"]=0.5


    layerwidth = 0.015
    dx = 0.075 - layerwidth / 2
    x = 0.5 + dx
    shift_amount = 0.2
    x += shift_amount
    buffer_distance = 0.05
    # Calculate midpoints of polygon

    right_vertices = xy[xy[:, 0] == np.max(xy[:, 0])]
    midpoint_right = np.mean(right_vertices, axis=0)

    arrow_start_x=midpoint_right[0]
    arrow_start = (arrow_start_x, 0.5) 
    arrow_end = (arrow_start_x+buffer_distance, 0.5)  
    ax.annotate("",  arrow_end,arrow_start, xycoords="axes fraction", arrowprops=rh_props)
    hidden_layer = Rectangle((x, 0.25), layerwidth, 0.5)
    ax.text(
        x + layerwidth / 2,
        0.75,
        "1024",
        **t,
    )
    
    

    x += dx
    r3height = 0.1
    out_layer = Rectangle((x, 0.5 - r3height / 2), layerwidth, r3height)
    ax.annotate("", (x, 0.5), (x - 0.08, 0.5), arrowprops=ph_props)
    ax.text(
        x + layerwidth / 2,
        0.55,
        "2",
        **t,
    )

    pc = PatchCollection([hidden_layer, out_layer], **bkwargs)
    t = txtkwargs.copy()
    t["usetex"] = False
    ax.text(
        x - dx,
        0.225,
        "",
        transform=ax.transAxes,
        **t,
        va="top",
    )
    ax.add_collection(pc)