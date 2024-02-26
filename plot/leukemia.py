import sys,os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../../'))
from tsimcne.evaluation.eval import knn_acc

file_root='../numpy_files'
file_names=['tsne_pixel_space.npy',
          'Pre-ResNet18.npy',
          'Y_AML_RC_HF_JI_GR.npy',
          'Y_AML_functrot90_RC_HF_VF_JI_GR.npy']
lbls=np.load(f'{file_root}/labels.npy')

def load_embed_(file_name,lbls):
    lbls=lbls.squeeze()
    all_embeddings=[]
    knn_acc_=[]
    for file in file_name:
        full_path=os.path.join(file_root,file)
        Y=np.load(full_path)
        knn_acc_.append(knn_acc(Y,lbls))
        all_embeddings.append(Y)
    return all_embeddings,lbls,knn_acc_



embed_list,lbls,knn_acc_= load_embed_(file_names,lbls)

im_index=[12,2,8,10,5,1]
label_color_dict = {
        0:('#2ca02c','BAS'), 
        1:("#008080",'EBO'),  
        2:("#808000",'EOS'),  
        3:("#FFFF00",'KSC'), 
        4:("#FF00FF",'LYA'), 
        5:('#ff7f0e','LYT'),
        6:("#800000",'MMZ'),  
        7:('#9467bd','MOB'), 
        8:('#8c564b','MON'), 
        9:('#e377c2','MYB'), 
        10:('#d62728','MYO'),
        11:('#7f7f7f','NGB'),  
        12:('#1f77b4','NGS'), 
        13:('#17becf','PMB'),  
        14:("#000000",'PMO')

}
def plot_diag(Y_i_list, new_labels, label_color_dict, title, col=2):
    num_plots = len(Y_i_list)
    rows = int(np.ceil(num_plots / col))

    rng = np.random.default_rng(303)
    shuf = rng.permutation(Y_i_list[0].shape[0])

    colors = [label_color_dict[label][0] for label in new_labels]
    colors_shuffled = np.array(colors)[shuf]

    fontprops = {'fontsize': "large", 
                 'fontweight': "bold"}

    markers_with_keys = [
        (key, mpl.lines.Line2D([], [], color=value[0], marker='o', linestyle='None', 
                            markersize=5, label=value[1])) 
        for key, value in label_color_dict.items() if key in im_index
    ]

    markers_sorted = sorted(markers_with_keys, key=lambda x: im_index.index(x[0]),reverse=True)

    markers = [marker for key, marker in markers_sorted]

    num = 0
    texts = ['a ', 'b ', 'c ', 'd '] 
    stylef='berenslab.mplstyle'

    with plt.style.context(stylef):
        fig, ax = plt.subplots(rows, col, figsize=(6, 2), squeeze=False, constrained_layout=True)
        for i in range(rows):
            for j in range(col):
                if num >= num_plots:
                    break
                data = Y_i_list[num]
                if num == 0:
                    xmin, xmax = ax[i, j].get_xlim()
                    ax[i, j].set_xlim(xmax, xmin)
                    ax[i, j].scatter(data[shuf][:, 0], data[shuf][:, 1], 
                                    c=colors_shuffled, rasterized=True)
                elif num ==2:
                    ax[i, j].scatter(data[shuf][:, 1], data[shuf][:, 0], 
                                    c=colors_shuffled, rasterized=True)
                    xmin, xmax = ax[i, j].get_ylim()
                    ax[i, j].set_ylim(xmax, xmin)
                else:
                    ax[i, j].scatter(data[shuf][:, 0], data[shuf][:, 1], 
                                    c=colors_shuffled, rasterized=True)
                
                ax[i, j].axis("equal")
                ax[i,j].set_title(f"{texts[num]}",loc="left",
                                        fontdict=fontprops)
                ax[i,j].set_title(f"$k$NN acc. = {title[num]}%",fontsize=7)
                ax[i, j].set_axis_off()
                num += 1
       


        fig.legend(handles=markers, 
                            ncol=1, loc="upper left", 
                            fontsize=7,bbox_to_anchor=(0.73, 0.75),frameon=False,
                            handletextpad=0.1, columnspacing=-0.1, borderaxespad=0)

    plt.savefig('../figures/fig2_leukemia.pdf',dpi=2000)
    plt.savefig('../figures/fig2_leukemia.png',dpi=2000)

plot_diag(Y_i_list=embed_list, new_labels=lbls,label_color_dict=label_color_dict,title=knn_acc_,col=4)