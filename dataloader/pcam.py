import os
import h5py
import gdown
import gzip
import shutil
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

def decompress_gzip(gz_root, output_path):
    with gzip.open(gz_root, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            print(f'Decompressed {gz_root}')
            
def download_data():
    train_labels={'url':'https://drive.google.com/uc?export=download&id=1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG','output':'train_lbls.h5.gz'}
    train={'url': 'https://drive.google.com/uc?export=download&id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2', 'output':'train.h5.gz'}
    valid={'url':'https://drive.google.com/uc?export=download&id=1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3','output':'valid.h5.gz'}
    valid_labels={'url':'https://drive.google.com/uc?export=download&id=1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO','output':'valid_lbls.h5.gz'}
    test={'url':'https://drive.google.com/uc?export=download&id=1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_','output':'test.h5.gz'}
    test_labels={'url':'https://drive.google.com/uc?export=download&id=17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP','output':'test_lbls.h5.gz'}
    
    root='../datasets/pcam16'  

    all_files=[train_labels,train,valid,valid_labels,test,test_labels]

    for files in all_files:
        output_filepath=f"{root}/{files['output']}"
        f=f"{files['output']}".split('.gz')[:-1][0]
        gdown.download(url=files['url'],output=output_filepath,fuzzy=True)
        decom_file_name=f"{root}/{f}"
        decompress_gzip(output_filepath, decom_file_name)

class PatchCamelyon(Dataset):
    def __init__(self, h5_dir, mode='train', transform=None, download=False):
        assert mode in ['train', 'valid', 'test']
        self.transform = transform
        self.h5_dir = h5_dir
        self.toImage = transforms.ToPILImage()    

        if download:
            download_data()

        self.x = h5py.File(os.path.join(h5_dir, 'train.h5'), 'r')
        self.y = h5py.File(os.path.join(h5_dir, 'train_lbls.h5'), 'r')
                        
    def __len__(self):
        return self.y['y'].shape[0]

    def __getitem__(self, idx):
        image = self.toImage(self.x['x'][idx])
        label =  torch.tensor(self.y['y'][idx][0][0][0], dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_all_label_numpy(self):
        return self.y['y'][:].squeeze()
    
    def __del__(self):
        self.x['x'].close()
        self.y['y'].close()

    

    