# from common import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
#from timm.models.resnet import *
from timm.models.efficientnet import efficientnet_b4, tf_efficientnet_b6, efficientnet_b2

from timm.models.efficientnet import *
from utils import BCEFocalLoss
import pdb
from nextvit import *
import torch.utils.checkpoint as checkpoint


class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]  #
    IMAGE_RGB_STD  = [0.5, 0.5, 0.5] #[0.229, 0.224, 0.225]  #

    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class Net(nn.Module):
    def load_pretrain(self, ):
        pass

    def __init__(self,):
        super(Net, self).__init__()
        self.output_type = ['inference', 'loss']

        # self.rgb = RGB()
        self.encoder = tf_efficientnet_b6(pretrained=True)
        #encoder_dim = [64, 256, 512, 1024, 2048]

        self.out = nn.Linear(2304,1)
        # self.out = nn.Conv1d(2304, 1, 1, 1)

        # self.uncertainty = nn.Conv2d(32,len(agbm_percentile), kernel_size=1)


    def forward(self, batch):
        # pdb.set_trace()
        x = batch['image']
        # x = x.expand(-1,3,-1,-1)
        # x = self.rgb(x) #17, 3, 256, 256
        x = (x - 0.5) / 0.5

        #------
        e = self.encoder
        # pdb.set_trace()
        x = e.forward_features(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = torch.flatten(x,1,3)
        #------

        feature = x
        # pdb.set_trace()
        out = self.out(feature)
        out = out.reshape(-1)

        output = {}
        if  'loss' in self.output_type:
            loss = F.binary_cross_entropy_with_logits(out, batch['label'])

            # print(batch['label'].shape)
            # focal_loss = BCEFocalLoss()(out, batch['label'])
            output['bce_loss'] = loss
            # output['focal_loss'] = focal_loss


        if 'inference' in self.output_type:
            out = torch.sigmoid(out)
            out = torch.nan_to_num(out)
            output['label'] = out

        return output



class NextViT(torch.nn.Module):
    def __init__(self, use_checkpoint, nextvit_checkpoint, *args, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        # self.with_aux_features = with_aux_features
        # self.with_aux_targets = with_aux_targets
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
        self.output_type = ['inference', 'loss']

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

        output = {}
        if  'loss' in self.output_type:
            loss = F.binary_cross_entropy_with_logits(out, batch['label'], reduction='none')
            # focal_loss = BCEFocalLoss()(out, batch['label'])
            output['bce_loss'] = loss
            # output['focal_loss'] = focal_loss


        if 'inference' in self.output_type:
            out = torch.sigmoid(out)
            out = torch.nan_to_num(out)
            output['label'] = out
        return output

# from nextvit_bn_merged import *

# class NextVitBNet(nn.Module):
#     def __init__(self,):
#         super(NextVitBNet, self).__init__()
#         self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
#         self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
#         self.encoder = nextvit_base(pretrained=False)

#     def forward(self, batch):
#         x = batch['image']

#         batch_size, C, H, W = x.shape
#         x = (x - self.mean) / self.std
#         # pdb.set_trace()
#         cancer = self.encoder(x)
#         cancer = cancer.reshape(-1)
#         # x = F.adaptive_avg_pool2d(e,1)
#         # cancer = torch.sigmoid(e)
#         output = {}
#         if  'loss' in self.output_type:
#             loss = F.binary_cross_entropy_with_logits(cancer, batch['cancer'])
#             # loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([49]).cuda())(cancer, batch['cancer'])
#             # loss = torch.nn.BCEWithLogitsLoss()(cancer, batch['cancer'])

#             focal_loss = BCEFocalLoss()(cancer, batch['cancer'])
#             output['bce_loss'] = loss
#             output['focal_loss'] = focal_loss


#         if 'inference' in self.output_type:
#             cancer = torch.sigmoid(cancer)
#             cancer = torch.nan_to_num(cancer)
#             output['cancer'] = cancer

#         return output

# class Net_b6(nn.Module):
#     def load_pretrain(self, ):
#         pass

#     def __init__(self,):
#         super(Net_b6, self).__init__()
#         self.output_type = ['inference', 'loss']

#         self.rgb = RGB()
#         self.encoder = tf_efficientnet_b6(pretrained=True)
#         #encoder_dim = [64, 256, 512, 1024, 2048]

#         # self.cancer = nn.Linear(1792,1)
#         self.cancer = nn.Linear(2304,1)

#         # self.uncertainty = nn.Conv2d(32,len(agbm_percentile), kernel_size=1)


#     def forward(self, batch):
#         # pdb.set_trace()
#         x = batch['image']
#         x = x.expand(-1,3,-1,-1)
#         x = self.rgb(x) #17, 3, 256, 256

#         #------
#         e = self.encoder
#         x = e.forward_features(x)
#         x = F.adaptive_avg_pool2d(x,1)
#         x = torch.flatten(x,1,3)
#         #------

#         feature = x
#         cancer = self.cancer(feature)
#         cancer = cancer.reshape(-1)

#         output = {}
#         if  'loss' in self.output_type:
#             # loss = F.binary_cross_entropy_with_logits(cancer, batch['cancer'])
#             # focal_loss = BCEFocalLoss()(cancer, batch['cancer'])
#             loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([49]).cuda())(cancer, batch['cancer'])
#             output['bce_loss'] = loss
#             # output['focal_loss'] = focal_loss


#         if 'inference' in self.output_type:
#             cancer = torch.sigmoid(cancer)
#             cancer = torch.nan_to_num(cancer)
#             output['cancer'] = cancer

#         return output

# # def run_check_net():

# #     h,w = 256, 256
# #     batch_size = 4

# #     # ---
# #     batch = {
# #         'image': torch.from_numpy(np.random.uniform(0,1,(batch_size,1,h,w))).float().cuda() ,
# #         'cancer': torch.from_numpy(np.random.choice(2,(batch_size))).float().cuda() ,
# #     }
# #     #batch = {k: v.cuda() for k, v in batch.items()}

# #     net = Net().cuda()
# #     # print(net)
# #     # torch.save({ 'state_dict': net.state_dict() },  'model.pth' )
# #     # exit(0)
# #     net.load_pretrain()

# #     with torch.no_grad():
# #         with torch.cuda.amp.autocast(enabled=True):
# #             output = net(batch)

# #     print('batch')
# #     for k, v in batch.items():
# #         print('%32s :' % k, v.shape)

# #     print('output')
# #     for k, v in output.items():
# #         if 'loss' not in k:
# #             print('%32s :' % k, v.shape)
# #     for k, v in output.items():
# #         if 'loss' in k:
# #             print('%32s :' % k, v.item())


# # # main #################################################################
# # if __name__ == '__main__':
# #     run_check_net()