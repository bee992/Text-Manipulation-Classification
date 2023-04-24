# from common import *
from augmentation import *
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Sampler
import torch
import pdb

image_size = 1024

def make_fold(df, fold=0):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    # train_df = df[df.patient_id.isin(train_id)].reset_index(drop=True)
    # valid_df = df[df.patient_id.isin(valid_id)].reset_index(drop=True)
    return train_df, valid_df


class build_dataset(Dataset):
    def __init__(self, df, train_val_flag=True, transforms=None):

        self.df = df
        self.train_val_flag = train_val_flag #
        self.img_paths = df['img_path'].tolist() 
        self.ids = df['img_name'].tolist()
        self.transforms = transforms

        if train_val_flag:
            self.label = df['img_label'].tolist()
        
    def __len__(self):
        return len(self.df)
        # return 8
    
    def __getitem__(self, index):
        #### id
        id       = self.ids[index]
        #### image
        img_path  = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [h, w, c]
        
        if self.train_val_flag: # train
            ### augmentations
            data  = self.transforms(image=img)
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            gt = self.label[index]
            # pdb.set_trace()
            return torch.tensor(img), torch.tensor(int(gt))
        
        else:  # test
            ### augmentations
            data  = self.transforms(image=img)
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            # pdb.set_trace()
            return torch.tensor(img), id

class DTTDataset(Dataset):
    def __init__(self, df, augment=None):
        self.df = df
        # self.train_val_flag = train_val_flag #
        self.img_paths = df['img_path'].tolist() 
        self.ids = df['img_name'].tolist()
        self.label = df['img_label'].tolist()
        self.augment = augment

    # def __str__(self):
    #     num_patient = len(set(self.df.patient_id))
    #     num_image = len(self.df)

    #     string = ''
    #     string += f'\tlen = {len(self)}\n'
    #     string += f'\tnum_patient = {num_patient}\n'
    #     string += f'\tnum_image = {num_image}\n'

    #     count = dict(self.df.cancer.value_counts())
    #     for k in [0,1]:
    #         string += f'\t\tcancer{k} = {count[k]:5d} ({count[k]/len(self.df):0.3f})\n'
    #     return string

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #### id
        id       = self.ids[index]
        #### image
        img_path  = self.img_paths[index] 
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [h, w, c]
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)/255
        # pdb.set_trace()
        # if self.train_val_flag: # train
            ### augmentations
        if self.augment != None:
            img  = self.augment(img)
        img = np.transpose(img, (2, 0, 1)) # [c, h, w]
        gt = self.label[index]
            # pdb.set_trace()
            
        
        # else:  # test
        #     ### augmentations
        #     data  = self.transforms(image=img)
        #     img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
        #     gt = id
        #     # pdb.set_trace()


        d = self.df.iloc[index]
        # # pdb.set_trace()
        # image = read_data(d)

        # if self.augment is not None:
        #     image = self.augment(image)

        r = {}
        r['index'] = index
        r['d'] = d
        # r['patient_id'] = d.patient_id  #
        r['image' ] = torch.from_numpy(img).float()
        r['label'] = torch.FloatTensor(gt)

        return r

tensor_key = ['image', 'label']
def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_key:
            v = torch.stack(v,0)
        d[k] = v
    # d['image']= d['image'].unsqueeze(1)
    d['label']= d['label'].reshape(-1)
    return d

class BalanceSampler(Sampler):

    def __init__(self, dataset, ratio=8):
        self.r = ratio-1
        self.dataset = dataset
        self.pos_index = np.where(dataset.df.img_label>0)[0]
        self.neg_index = np.where(dataset.df.img_label==0)[0]

        self.length = self.r*int(np.floor(len(self.neg_index)/self.r))

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[:self.length].reshape(-1,self.r)
        pos_index = np.random.choice(pos_index, self.length//self.r).reshape(-1,1)

        index = np.concatenate([pos_index,neg_index],-1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.length

#################################################################################

def train_augment_v00a(image):
    image = do_random_hflip(image) # hflip, vflip or both
    #image, target = do_random_hflip(image, target)

    if np.random.rand() < 0.2:
        for func in np.random.choice([
            lambda image : do_random_affine( image, degree=15, translate=0.1, scale=0.2, shear=10),
            lambda image : do_random_rotate(image,  degree=15),
            lambda image : do_random_stretch(image, stretch=(0.2,0.2)),
        ], 1):
            image = func(image)

    if np.random.rand() < 0.1:
        image = do_elastic_transform(
            image,
            alpha=image_size,
            sigma=image_size* 0.05,
            alpha_affine=image_size* 0.03
        )

    if np.random.rand() < 0.2:
        for func in np.random.choice([
            lambda image: do_random_contrast(image),
        ], 1):
            image = func(image)
            pass

    return image

def train_augment_v00(image):
    image = do_random_hflip(image) # hflip, vflip or both
    #image, target = do_random_hflip(image, target)

    if np.random.rand() < 0.7:
        for func in np.random.choice([
            lambda image : do_random_affine( image, degree=30, translate=0.1, scale=0.3, shear=20),
            lambda image : do_random_rotate(image,  degree=30),
            lambda image : do_random_stretch(image, stretch=(0.3,0.3)),
        ], 1):
            image = func(image)

    if np.random.rand() < 0.25:
        image = do_elastic_transform(
            image,
            alpha=image_size,
            sigma=image_size* 0.05,
            alpha_affine=image_size* 0.03
        )
    if np.random.rand() < 0.25:
        image = do_random_cutout(
            image, num_block=5,
            block_size=[0.1,0.3],
            fill='constant'
        )

    if np.random.rand() < 0.5:
        for func in np.random.choice([
            lambda image: do_random_contrast(image),
            lambda image: do_random_noise(image, m=0.1),
        ], 1):
            image = func(image)
            pass

    return image

#################################################################################

def run_check_dataset():
    train_df, valid_df = make_fold()
    dataset = RsnaDataset(train_df, augment=train_augment_v00)
    print(dataset)

    for i in range(100):
        i = 0 #240*8+ i#np.random.choice(len(dataset))
        r = dataset[i]
        print(r['index'], 'id = ', r['patient_id'], '-----------')
        for k in tensor_key :
            v = r[k]
            print(k)
            print('\t', 'dtype:', v.dtype)
            print('\t', 'shape:', v.shape)
            if len(v)!=0:
                print('\t', 'min/max:', v.min().item(),'/', v.max().item())
                print('\t', 'is_contiguous:', v.is_contiguous())
                print('\t', 'values:')
                print('\t\t', v.reshape(-1)[:8].data.numpy().tolist(), '...')
                print('\t\t', v.reshape(-1)[-8:].data.numpy().tolist())
        print('')
        if 1:
            image  = r['image'].data.cpu().numpy()

            # image_show_norm('image', image)
            cv2.waitKey(0)


    loader = DataLoader(
        dataset,
        #sampler=SequentialSampler(dataset),
        sampler=BalanceSampler(dataset),
        batch_size=8,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate,
    )
    print(loader.batch_size, len(loader), len(dataset))
    print('')

    for t, batch in enumerate(loader):
        if t > 5: break
        print('batch ', t, '===================')
        print('index', batch['index'])
        for k in tensor_key:
            v = batch[k]
            print(k)
            print('\t', 'shape:', v.shape)
            print('\t', 'dtype:', v.dtype)
            print('\t', 'is_contiguous:', v.is_contiguous())
            print('\t', 'value:')
            print('\t\t', v.reshape(-1)[:8].data.numpy().tolist())
            if k=='cancer':
                print('\t\tsum ', v.sum().item())

        print('')

def run_check_augment():

    train_df, valid_df = make_fold()
    dataset = RsnaDataset(train_df)
    print(dataset)

    #---------------------------------------------------------------
    def augment(image):
        # image, target = do_random_hflip(image, target)
        #image, target = do_random_flip(image, target)

        #image, target = do_random_affine( image, target, degree=10, translate=0.1, scale=0.2, shear=10)
        #image, target = do_random_rotate(image, target, degree=45)
        #image, target = do_random_rotate90(image, target)

        #image, target = do_random_perspective(image, target, m=0.3)
        #image, target = do_random_zoom_small(image, target)

        #image = do_random_hsv(image, h=20, s=50, v=50)
        # image = do_random_contrast(image)
        # image = do_random_gray(image)
        # image = do_random_guassian_blur(image, k=[3, 5], s=[0.1, 2.0])
        # image = do_random_noise(image, m=0.08)
        return image

    for i in range(10):
        #i = 2424 #np.random.choice(len(dataset))#272 #2627
        print(i)
        r = dataset[i]

        image  = r['image'].data.cpu().numpy()
        # image_show_norm('image',image, min=0, max=1,resize=1)
        #cv2.waitKey(0)

        for t in range(100):
            #image1 = augment(image.copy())
            image1 = train_augment_v00(image.copy())
            # image_show_norm('image1', image1, min=0, max=1,resize=1)
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    run_check_dataset()
    #run_check_augment()