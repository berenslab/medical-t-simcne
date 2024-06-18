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

root_="../numpy_files"

train = medmnist.PathMNIST(
    "train", root=root_, download=True
)
test = medmnist.PathMNIST(
    "test", root=root_, download=True
)
val = medmnist.PathMNIST(
    "val", root=root_, download=True
)
dataset = torch.utils.data.ConcatDataset([train, test, val])

subclusters={'adipose':(92, 248,40, 358),
             'background':(271,157,421, 227),
             'debris':(-40,-143,139, -140),
             'lymphocytes':(-151,-119,-291, -109),
             'mucus':(-29,105,-220, 305),
             'smooth muscle':(96,15,226, -49.7),#356, 95
             'colon mucosa': (-117,47,-280, 107),
             'cancer-ass. stroma':(38,6.6,356, 95),#278, 22
             'col-adenocarcinoma':(-27,-28,-240, 187),
             'artifacts2':(-3,-170,137, -203),
             'colon mucosa2':(-120,-15,-280, 45),
             'debris2':(-78,-148,-58, -225),
             'col-adenocarcinoma2':(-70,-73,-270, -23),
             'background2':(223,206,363, 336),
             'smooth muscle2':(106,-51,226, -49.7),
             'mucus 2':(11,100,-30, 238 ),
             'lymphocytes2':(-157,-152,-297, -177),
             'adipose2':(104,225,204, 345)
             }

def annotate_path(ax, Y, dataset, arrowprops=None):
    labels = np.load(f"{root_}/label_PathMNIST.npy")

    if arrowprops is None:
        arrowprops = dict(
            arrowstyle="-",
            linewidth=plt.rcParams["axes.linewidth"],
            color="xkcd:slate gray",
        )

    rng = np.random.default_rng(56199999)
    n_samples = 3
    knn = NearestNeighbors(n_neighbors=15, n_jobs=8)
    knn.fit(Y)

    norm = Normalize(vmin=min(labels), vmax=max(labels))
    cm = plt.get_cmap('tab10',lut=len(np.unique(labels)))
    

    for key, (x, y, x_axis,y_axis) in subclusters.items():
        img_idx = knn.kneighbors([[x, y]], return_distance=False).squeeze()
        imgs = []
        for ix in rng.choice(img_idx, size=n_samples, replace=False):
            im,lbl= dataset[ix]
            lbl = int(labels[ix])
            im=np.array(im)
            pil_img = Image.fromarray(im)

            border_color = cm(norm(lbl))

            border_color = tuple(int(c * 255) for c in border_color)

            pil_img_with_border = add_border(pil_img, border_size=2, border_color=border_color)

            img_with_border = np.array(pil_img_with_border)
            imbox = mpl.offsetbox.OffsetImage(img_with_border, zoom=0.2)
            imgs.append(imbox)

        imrow = mpl.offsetbox.HPacker(children=imgs, pad=0, sep=2)
        
        if not re.search(r'\d', key):
            txt = mpl.offsetbox.TextArea(key.capitalize(),
                                          textprops=dict(size=4))
            annot = mpl.offsetbox.VPacker(
                children=[txt, imrow], sep=1, align="center")
            
        else:
            txt=mpl.offsetbox.TextArea('')
            annot = mpl.offsetbox.VPacker(
                children=[imrow], sep = 1, align="center"
            )
            
        abox = mpl.offsetbox.AnnotationBbox(
            annot,
            (x, y),
            (x_axis+20,y_axis+20) ,
            arrowprops=arrowprops,
            frameon=False)
        ax.add_artist(abox)

