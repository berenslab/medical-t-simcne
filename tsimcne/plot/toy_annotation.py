import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, HPacker, VPacker
from PIL import Image, ImageOps
from matplotlib.colors import ListedColormap

root='../../numpy_files'
def add_border(input_image, border_size, border_color):
    img_with_border = ImageOps.expand(input_image, border=border_size, fill=border_color)
    return img_with_border

def plot_embeddings(classes,im_index,npz_path,ax, cm, title,representative_indices,box_x):
    npz = np.load(npz_path)
    Y = npz['embeddings']
    labels = npz['labels']
    images_ = npz['images']

    if len(labels)==10015: #we binarize the labels
        labels=np.where(labels==5,1,0)

    rng = np.random.default_rng(303)
    shuf = rng.permutation(Y.shape[0])
    ax.scatter(Y[shuf][:, 0], Y[shuf][:, 1], c=labels[shuf], s=1, alpha=0.75, rasterized=True, cmap=cm)
    ax.set_xlim([min(Y[:,0]) - 1, max(Y[:,0]) + 1])
    ax.axis("equal")
    ax.axis("off")

    x_coords = np.repeat(np.linspace(0, 1, len(representative_indices) // 2, endpoint=False), 2)/30
    x_coords[::2] += 0.45/30

    coord_tuples = [(x, idx, Y[idx]) for x, idx in zip(x_coords, representative_indices)] 
    coord_tuples.sort(key=lambda tup: tup[2][1])

    
    annotations=[]

    norm = Normalize(vmin=min(labels), vmax=max(labels))

    for i, (x, idx, xy) in enumerate(coord_tuples):
        img = images_[idx]
        lbl = int(labels[idx])

        pil_img = Image.fromarray(img)

        border_color = cm(norm(lbl))

        border_color = tuple(int(c * 255) for c in border_color)

        pil_img_with_border = add_border(pil_img, border_size=2, border_color=border_color)

        img_with_border = np.array(pil_img_with_border)

        imbox = OffsetImage(img_with_border, zoom=0.25)
        annotations.append((x, xy, lbl, imbox))

    annotations.sort(key=lambda tup: tup[1][1])
    
    num_labels = len(im_index)
    vertical_spacing = np.linspace(0, 1, num_labels + 2)[1:]
    label_vertical_positions = dict(zip(im_index, vertical_spacing))


    points_by_label = {label: [] for label in im_index }
    for x,xy, label, imbox in annotations:
        points_by_label[label].append((xy, imbox))

    for idx, (label, items) in enumerate(points_by_label.items()):
        if not items:
            continue

        common_x = np.mean([xy[0] for xy, _ in items])
        common_y = np.mean([xy[1] for xy, _ in items])

        textprops = {"fontsize": 4}
        imgs = [imbox for _, imbox in items]
        imrow = HPacker(children=imgs, pad=0, sep=1.5)
        txt = TextArea(classes[label],textprops=textprops)
        annot = VPacker(children=[txt, imrow], pad=0, sep=1, align="center")
        box_y = label_vertical_positions[label] 
        abox = AnnotationBbox(annot, 
                                (common_x,common_y),
                                xybox=(box_x, box_y), 
                                xycoords='data',
                                boxcoords='axes fraction',
                                frameon=False,
                                box_alignment=(0, 0)
                                                )
        ax.add_artist(abox)
    ax.set_title(title[0],loc='left',fontweight='bold')
    ax.set_title(title[1])

def annotate_aml(ax):
    npz_path = f'{root}/aml_new_default.npz'
    label_color_dict = {
        0: ('#2ca02c', 'BAS'),
        1: ("#008080", 'EBO'),
        2: ("#808000", 'EOS'),
        3: ("#FFFF00", 'KSC'),
        4: ("#FF00FF", 'LYA'),
        5: ('#ff7f0e', 'LYT'),
        6: ("#800000", 'MMZ'),
        7: ('#9467bd', 'MOB'),
        8: ('#8c564b', 'MON'),
        9: ('#e377c2', 'MYB'),
        10: ('#d62728', 'MYO'),
        11: ('#7f7f7f', 'NGB'),
        12: ('#1f77b4', 'NGS'),
        13: ('#17becf', 'PMB'),
        14: ("#000000", 'PMO')
    }
    cm = ListedColormap([color for color, _ in label_color_dict.values()])
    good_indices=[(1, 80),(1, 93),(1, 155),
                  (2, 408),(2, 474), (2, 352),
                  (5, 1616),(5, 1782),(5, 3685),
                  (8, 4805), (8, 5594),(8, 5268),
                  (10, 7762),(10, 7142),(10,8733),
                  (12, 12887),(12, 17003),(12,9863)]
    representative_indices=[val[1] for val in good_indices]
    classes=[]
    for label in label_color_dict.keys():
        classes.append(label_color_dict[label][1])
    im_index = [12, 2, 8, 10, 5, 1]
    plot_embeddings(classes,im_index,npz_path,ax, cm, ['a','Leukemia'],representative_indices,0.8)

def annotate_bld(ax):
    classes={
            6: 'NEU',
            2: 'EBO',
            7: 'PLA',
            1: 'EOS',
            5: 'MONO',
            3: 'GRA',
            4: 'LYMPH',
            0: 'BAS'}
    npz_path=f'{root}/bloodmnist_med_def2.npz'
    cm = plt.get_cmap('tab10', lut=8)
    im_index=[0,4,3,5,1,7,2,6]
  

    good_indices=[(0, 3848), (1, 7338), (2, 1365),(3, 16017),(4, 10888),(5, 2230),(6, 3715),(7, 1108),
    (0, 8620), (1, 6903), (2, 16612), (3, 9224), (4, 6816), (5, 2344), (6, 8305), (7, 11490),
    (0, 10425), (1, 9772), (2, 1981), (3, 7511), (4, 1689), (5, 16348), (6, 10957), (7, 5368)]
    representative_indices=[val[1] for val in good_indices]
    plot_embeddings(classes,im_index,npz_path,ax, cm, ['b','BloodMNIST'],representative_indices,0.95)

def annotate_derma(ax):
    classes={0: 'OTHERS',
             1: 'M.NEVI'}
    npz_path=f'{root}/dermamnist_data2.npz'
    
    cm = ListedColormap(["#ff0000","#1f77b4"])
    im_index=[0,1]
    good_indices=[(0, 872), (0, 64), (0, 5190), (1, 632), (1, 9920), (1, 8043)]
    representative_indices=[val[1] for val in good_indices]
    plot_embeddings(classes,im_index,npz_path,ax, cm, ['c','DermaMNIST'],representative_indices,0.8)

def main():
    stylef = "berenslab.mplstyle"
    grid_rows, grid_cols = 1, 3
    fig = plt.figure(figsize=(6, 2.2))
    fig.tight_layout()

    all_axes = []
 
    datasets_info = [
        {"func": annotate_aml},
        {"func": annotate_bld},
        {"func": annotate_derma}
    ]

    for i, dataset in enumerate(datasets_info):
        ax = plt.subplot2grid((grid_rows, grid_cols), (0, i), rowspan=grid_rows, colspan=grid_cols // 3, fig=fig)
        with plt.style.context(stylef):
            dataset['func'](ax)
        all_axes.append(ax)

    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('../../figures/toy_datasets.png', dpi=1000)
    plt.savefig('../../figures/toy_datasets.pdf', dpi=1000)


if __name__ == "__main__":
    main()
