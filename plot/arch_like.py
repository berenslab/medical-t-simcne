import random
random.seed(10)
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
warnings.filterwarnings('ignore')
import torchvision.transforms.functional as F
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from torchvision.transforms.functional import to_pil_image

def cropfun(t):
    random.seed(10)
    scale = (0.08, 1.0)
    ratio = (0.75, 1.3333333333333333)
    image_height,image_width = 28,28
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
def cjfun(t):
    t = F.adjust_brightness(t, 0.5)
    t = F.adjust_contrast(t, 0.4)
    t = F.adjust_saturation(t, 0.1)
    t = F.adjust_hue(t, 0.2)
    return t

def draw_arch(ax, orig_im,  zoom = 0.65):
    # torch.manual_seed(rng.integers(2**64, dtype="uint"))
    plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

    ax.set_axis_off()   

    aprops = dict(
        arrowstyle="-|>",
        color="xkcd:dark gray",
        linewidth=plt.rcParams["axes.linewidth"],
        relpos = (1, 0.5),
        shrinkB = 0.1,
    
    )
    txtkwargs = dict(
        usetex=True,
        horizontalalignment="center",
        verticalalignment="center",
    )

    
    t_im1 = cropfun(orig_im)
    t_im2 = F.vflip(orig_im)

    new_props = aprops.copy()
    new_props['mutation_scale'] = 3
    new_props['shrinkB'] = 15  #determines the length of the arrow


    xybox_cord=(-0.3, 0.47)
    im0 = OffsetImage(to_pil_image(orig_im), zoom = zoom)
    y_mid = get_image_midpoint(t_im1, xybox_cord, zoom)

    abox0 = AnnotationBbox(
        im0,
        xy = (0.01, y_mid[1]),  #arrow
        xybox = xybox_cord, #image
        xycoords = "axes fraction",
        frameon = False,
        arrowprops = new_props,
        
    )
    ax.add_artist(abox0)

    augprops = dict(
            arrowstyle="<|-|>",
            color="xkcd:dark gray",
            linewidth=plt.rcParams["axes.linewidth"],
            connectionstyle="bar,fraction=0.3",
            mutation_scale = 3,
        )
    

    arrow_length = 0.2
    start_point_y = y_mid[1] - arrow_length  
    end_point_y = y_mid[1] + arrow_length   

    ax.annotate(
        "",
        (0.08, start_point_y),  # start of two-edged arrow
        (0.08, end_point_y),  # end of two-edged arrow
        xycoords="axes fraction",
        arrowprops = augprops,
    )

    # img1 
    se_props = aprops.copy()
    se_props['mutation_scale'] = 3
    images_x = 0.15
    abox1_xybox=(images_x, end_point_y)

    im1 = OffsetImage(to_pil_image(t_im1), zoom=zoom)
    y_mid1=get_image_midpoint(t_im1,abox1_xybox,zoom)
    
    b4res_arrow_x=0.3
    abox1 = AnnotationBbox(
        im1,
        xy=(b4res_arrow_x, y_mid1[1]), 
        xybox=abox1_xybox, #image
        xycoords="axes fraction",
        frameon=False,
        arrowprops=se_props, #up big arrow
    )
    ax.add_artist(abox1)
    

    # img2 
    im2 = OffsetImage(to_pil_image(t_im2), zoom=zoom)
    abox2_xybox=(images_x, start_point_y)
    y_mid2=get_image_midpoint(t_im2,abox2_xybox,zoom)
    abox2 = AnnotationBbox(
        im2,
        xy=(b4res_arrow_x, y_mid2[1]),  
        xybox=abox2_xybox, 
        xycoords="axes fraction",
        frameon=False,
        arrowprops=se_props, #down big arrow
    )
    ax.add_artist(abox2)
    

    t = txtkwargs.copy()

    t["usetex"] = False
    t["fontsize"] = 5
    t["rotation"] = 90
    t["transform"] = ax.transAxes
    t["rotation_mode"] = "anchor"
    t["horizontalalignment"] = "right"
    
    text_x_near_abox1 = abox1_xybox[0]  #data augmentation top horizontal position
    text_y_near_abox1 = abox1_xybox[1]
 
    ax.text(
            text_x_near_abox1 - 0.2, 
            0.528,
            s="data",
            va="bottom",
            **t,
        )
    ax.text(
        text_x_near_abox1 - 0.165,
        text_y_near_abox1 + 0.009 ,
        s="augnmentation",
        va="bottom",
        **t,
    )

    bkwargs = dict(
        edgecolor="xkcd:slate gray",
        facecolor="xkcd:white",
        linewidth=plt.rcParams["axes.linewidth"],
    )
    # resnet polygon
    initial_adjustment_x = 0.05  # move polygon left
    initial_adjustment_y = -0.0455 # move polygon up

    x_vertex=[0.2,  0.2,  0.55, 0.55]
    y_vertex=[-0.16, 1.2, 0.92, 0.17]  
    xy=np.array(list(zip(x_vertex,y_vertex)))
    
    xy[:, 0] += initial_adjustment_x
    xy[:, 1] += initial_adjustment_y

    centroid = np.mean(xy, axis=0)
    scaling_factor = 0.65
    xy = centroid + (xy - centroid) * scaling_factor

    resnet = Polygon(xy, closed=True, **bkwargs)

    t = txtkwargs.copy()
    t["usetex"] = False
    ax.text(*xy.mean(axis=0),
             " ResNet \n + \n proj. head ", **t)
    ax.add_artist(resnet)
  
    
    ph_props = aprops.copy()
    ph_props["shrinkB"] = 1
    ph_props["shrinkA"] = 5
   
    x = 0.685
    r3height = 0.4
    square_start_y = 0.5 - r3height / 2
    square_height = square_start_y +  r3height
    width = square_height - 0.45
    out_layer = Rectangle((x, square_start_y), width, r3height)
    rect_x, rect_y = out_layer.get_xy() 
    rect_width = out_layer.get_width()  
    rect_height = out_layer.get_height()  

    text_x = rect_x + 0.01 + rect_width / 2  
    text_y = rect_y + rect_height + 0.04 

    square_t = t.copy()

    square_t["usetex"] = False
    square_t["fontsize"] = 7
    ax.text(
        text_x,
        text_y,
        r"$\mathbb{R}^{2}$",
        **square_t,
    )
    pc = PatchCollection([ out_layer], **bkwargs)
    t = txtkwargs.copy()
    t["usetex"] = False

    ax.add_collection(pc)
#####################################################
    right_vertices = xy[xy[:, 0] == np.max(xy[:, 0])]
    midpoint_right = np.mean(right_vertices, axis=0)
    buffer_distance = 0.1
    arrow_start_x = midpoint_right[0]

    arrow_start_y_d = 0.4  
    arrow_end_x_d = arrow_start_x + buffer_distance + 0.2
    arrow_end_y_d = arrow_start_y_d
    crest_d = (arrow_start_x + (arrow_end_x_d - arrow_start_x) / 3, arrow_start_y_d)
    trough_d = (arrow_start_x + 2 * (arrow_end_x_d - arrow_start_x) / 3, arrow_start_y_d - 0.1)
    verts_d = [(arrow_start_x, arrow_start_y_d), crest_d, trough_d, (arrow_end_x_d, arrow_end_y_d)]
    codes_d = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path_d = Path(verts_d, codes_d)
    arrow_patch_d = FancyArrowPatch(path=path_d, arrowstyle='-|>',
                    color="xkcd:dark gray",
                    mutation_scale = 3,
                    linewidth=plt.rcParams["axes.linewidth"])
    ax.add_patch(arrow_patch_d)

    
    # arrow_start_x =    midpoint_right[0]
    arrow_start_y_u = 0.6  
    arrow_end_x_u = arrow_start_x + buffer_distance + 0.1 
    arrow_end_y_u = arrow_start_y_u 
    
    arrow_end_x_u = arrow_start_x + buffer_distance + 0.1 
    arrow_end_y_u = arrow_start_y_u  

    mid_control_x_u = arrow_start_x + (arrow_end_x_u - arrow_start_x) / 2
    mid_control_y_u = arrow_start_y_u + 0.03  

    verts_u = [
        (arrow_start_x, arrow_start_y_u),  # Start 
        (mid_control_x_u, mid_control_y_u),  # Mid control for upward arc
        (arrow_end_x_u, arrow_end_y_u)  # End 
    ]
    # Bezier curve
    codes_u = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path_u = Path(verts_u, codes_u)
    arrow_patch_u = FancyArrowPatch(path=path_u, arrowstyle='-|>',   
                    color="xkcd:dark gray",
                    mutation_scale = 3,
                    linewidth=plt.rcParams["axes.linewidth"]
           )
    ax.add_patch(arrow_patch_u)

    #####################################################

    ax.text(arrow_end_x_u+0.015, arrow_end_y_u+0.028, '$z_{1}$', **t)
    ax.text(arrow_end_x_d+0.028, arrow_end_y_d-0.01, '$z_{2}$', **t)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    aspect = ax.get_aspect()

    maker_end_x_u=arrow_end_x_u+0.007
    maker_end_y_d=arrow_end_y_d+0.007
    marker_end_x_d=arrow_end_x_d+0.005

    ax.plot([maker_end_x_u, marker_end_x_d],
     [arrow_end_y_u, arrow_end_y_d],
     linestyle='--',
     color='xkcd:slate gray',
     )
    
    ax.plot(maker_end_x_u, arrow_end_y_u, marker="o", markersize=1, markeredgecolor="black", markerfacecolor="black")
    ax.plot(marker_end_x_d, maker_end_y_d, marker="o", markersize=1, markeredgecolor="black", markerfacecolor="black")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(aspect)

