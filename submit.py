import pdb
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from timm.models.efficientnet import efficientnet_b7, efficientnet_b6, efficientnet_b4, efficientnet_b2, tf_efficientnet_b6
from timeit import default_timer as timer
from nextvit import *
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm

image_size = 1024


def create_df_test(path_test):
    col_name = ['img_name', 'img_path']
    imgs_info = []
    for img_name in sorted(os.listdir(path_test)):
        if img_name.endswith('.jpg'):
            imgs_info.append([img_name, os.path.join(path_test, img_name)])

    imgs_info_array = np.array(imgs_info)
    df = pd.DataFrame(imgs_info_array, columns=col_name)
    df.to_csv('test.csv')

def read_image(d):
    image = cv2.imread(f'{d.img_path}', cv2.IMREAD_COLOR)
    return image

class Dataset(Dataset):
    def __init__(self, df):
        
        self.df = df
        self.ids = df['img_name'].tolist()


    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, index):
        d = self.df.iloc[index]
        id = self.ids[index]
        image = read_image(d)

        image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)

        image = image.astype(np.float32) / 255.0

        r = {}
        r['index'] = index
        r['d'] = d
        r['id'] = id
        r['name'] = d.img_name
        r['image'] = image_to_tensor(image).float()
        # r['image'] = torch.from_numpy(image)
        # r['label'] = torch.FloatTensor([d.img_label])
        # pdb.set_trace()
        return r

tensor_key = ['image']

def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_key:
            v = torch.stack(v, 0)
        d[k] = v
    # d['image'] = d['image'].unsqueeze(1)
    # d['label'] = d['label'].reshape(-1)
    return d

def image_to_tensor(image, mode='bgr'): #image mode
	if mode=='bgr':
		image = image[:,:,::-1]
	x = image
	x = x.transpose(2,0,1)
	x = np.ascontiguousarray(x)
	x = torch.tensor(x, dtype=torch.float)
	return x




class EffNet(nn.Module):
    def load_pretrain(self, ):
        pass

    def __init__(self,):
        super(EffNet, self).__init__()
        self.encoder = tf_efficientnet_b6(pretrained=False)

        self.out = nn.Linear(2304,1)


    def forward(self, batch):
        x = batch['image']
        x = (x - 0.5) / 0.5
        #------
        e = self.encoder
        # pdb.set_trace()
        x = e.forward_features(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = torch.flatten(x,1,3)
        #------

        feature = x
        out = self.out(feature)
        out = out.reshape(-1)
        out = torch.sigmoid(out)
        out = torch.nan_to_num(out)
        return out

class NextViT(torch.nn.Module):
    def __init__(self, use_checkpoint, nextvit_checkpoint, *args, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        if "base" in nextvit_checkpoint:
            model = nextvit_base()
        elif "small" in nextvit_checkpoint:
            model = nextvit_small()
        else:
            raise RuntimeError()
        model.load_state_dict(torch.load(nextvit_checkpoint)["model"])
        self.stem = model.stem
        self.features = model.features
        self.norm = model.norm
        self.avgpool = model.avgpool
        self.proj_head = torch.nn.Linear(1024, 1)

    def forward(self, batch):
        x = batch['image']
        x = (x - self.mean) / self.std
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.proj_head(x)
        out = out.reshape(-1)
        out = torch.sigmoid(out)
        out = torch.nan_to_num(out)
        return out


def off_submit():

    model = [
        # 1024
   
        # # 0313 提交
        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-0/checkpoint/18.000000000002974.model.pth'],
        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-0/checkpoint/19.000000000003194.model.pth'],

        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-1/checkpoint/14.000000000002093.model.pth'],
        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-1/checkpoint/16.000000000002533.model.pth'],

        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-2/checkpoint/18.000000000002974.model.pth'],
        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-2/checkpoint/19.000000000003194.model.pth'],

        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-3/checkpoint/15.000000000002313.model.pth'],
        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-3/checkpoint/16.000000000002533.model.pth'],

        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-4/checkpoint/17.000000000002753.model.pth'],
        [EffNet, '../user_data/checkpoints/effb6-1024-v00a-RandomSampler-StepLR/fold-4/checkpoint/18.000000000002974.model.pth'],

        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-0/checkpoint/29.000000000003094.model.pth'],
        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-0/checkpoint/35.00000000000403.model.pth'],

        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-1/checkpoint/25.00000000000247.model.pth'],
        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-1/checkpoint/18.000000000001375.model.pth'],

        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-2/checkpoint/36.000000000004185.model.pth'],
        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-2/checkpoint/37.00000000000434.model.pth'],

        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-3/checkpoint/29.000000000003094.model.pth'],
        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-3/checkpoint/30.00000000000325.model.pth'],

        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-4/checkpoint/38.0000000000045.model.pth'],
        [NextViT, '../user_data/checkpoints/NextViT-1024-v00a-RandomSampler-CosineLR-lr2e4/fold-4/checkpoint/39.000000000004654.model.pth'],
        
       

    ]

    num_net = len(model)
    print(f'num_net = {num_net}')

    net = []
    for i in range(num_net):
        Net, checkpoint = model[i]
        if 'NextViT' in checkpoint:
            n = NextViT(False, '../user_data/nextvit_pretrained/nextvit_small_in1k_384.pth')
        else:
            n = Net()
        f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        n.load_state_dict({k.replace('module.',''):v for k,v in f['state_dict'].items()}, strict=True)  # True
        n.cuda()
        n.eval()
        net.append(n)

    test_df = pd.read_csv('test.csv')
    test_dataset = Dataset(test_df)

    test_loader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size  = 6,
        drop_last   = False,
        num_workers = 8,     
        pin_memory  = True,
        collate_fn = null_collate,
    )

    #----
    if 1:
        result = {
            'i':[],
            'probability':[],
        }
        test_num = 0

        start_timer = timer()
        for t, batch in enumerate(tqdm(test_loader)):
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].cuda()

            p = 0
            count = 0
            with torch.no_grad():
                with amp.autocast(enabled=True):
                    for i in range(num_net):
                        p += net[i](batch)
                        count += 1

                        # TTA
                        batch['image'] = torch.flip(batch['image'], dims=[3, ])
                        p += net[i](batch)
                        count += 1

            p = p / count
            test_num += batch_size
            test_df.loc[test_df['img_name'].isin(batch['id']), 'pred_prob'] = p.float().data.cpu().numpy()
        print('Done')
    
    submid_df = test_df.loc[:,['img_name', 'pred_prob']]
    submid_df.to_csv(f"../prediction_result/submission.csv", header=False, index=False, sep=' ')


if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    create_df_test('../tianchi_data/data/test')
    off_submit()