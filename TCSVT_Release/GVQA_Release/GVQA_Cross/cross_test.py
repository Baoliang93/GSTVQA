#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 03:26:04 2020

@author: blchen6
"""


#test_index= [17, 37, 40, 46, 52, 67, 70, 77, 99, 130, 140, 145, 147, 164, 173, 201, 202, 225, 231, 242, 248, 254, 262, 301, 302, 322, 328, 337, 350, 378, 425, 426, 435, 441, 444, 454, 458, 479, 481, 488, 498, 507, 533, 534, 546, 570, 574, 585, 601, 604, 613, 618, 635, 642, 644, 668, 696, 705, 731, 736, 742, 746, 768, 775, 782, 795, 798, 801, 807, 817, 820, 838, 864, 873, 875, 882, 884, 887, 891, 892, 903, 904, 913, 916, 945, 950, 973, 975, 985, 997, 1008, 1010, 1022, 1032, 1033, 1047, 1052, 1069, 1076, 1085, 1086, 1089, 1093, 1099, 1105, 1122, 1131, 1139, 1156, 1161, 1162, 1184, 1191, 1200, 1201, 1215, 1217, 1227, 1231, 1249, 1252, 1275, 1289, 1297, 1305, 1321, 1322, 1332, 1337, 1353, 1394, 1399, 1418, 1438, 1441, 1451, 1452, 1454, 1462, 1469, 1476, 1481, 1484, 1490, 1493, 1516, 1519, 1525, 1531, 1556, 1559, 1589, 1602, 1614, 1643, 1648, 1650, 1653, 1656, 1672, 1678, 1695, 1704, 1716, 1717, 1727, 1732, 1746, 1749, 1756, 1776, 1781, 1786, 1790, 1794, 1797, 1808, 1815, 1827, 1828, 1835, 1843, 1857, 1859, 1868, 1871, 1877, 1878, 1884, 1886, 1909, 1910, 1944, 1946, 1952, 1964, 1971, 1976, 1979, 1992, 1995, 2003, 2004, 2037, 2042, 2052, 2057, 2076, 2077, 2087, 2096, 2111, 2124, 2135, 2145, 2174, 2192, 2198, 2202, 2209, 2221, 2226, 2231, 2241, 2242, 2253, 2260, 2262, 2266, 2305, 2334, 2346, 2349, 2361, 2382, 2417, 2421, 2442, 2453, 2487, 2514, 2525, 2535, 2537, 2549, 2555, 2597, 2606, 2626, 2632, 2653, 2657, 2669, 2675, 2694, 2701, 2708, 2714, 2721, 2725, 2728, 2733, 2738, 2744, 2756, 2778, 2795, 2804, 2811, 2812, 2814, 2830, 2831, 2835, 2877, 2890, 2894, 2903, 2905, 2917, 2951, 2956, 2975, 2991, 3002, 3009, 3030, 3038, 3045, 3059, 3079, 3081, 3083, 3093, 3103, 3113, 3116, 3117, 3118, 3119, 3130, 3138, 3139, 3157, 3164, 3169, 3173, 3176, 3183, 3207, 3217, 3225, 3227, 3230, 3250, 3295, 3296, 3300, 3323, 3344, 3351, 3374, 3376, 3377, 3384, 3389, 3410, 3426, 3428, 3429, 3431, 3433, 3440, 3444, 3465, 3471, 3475, 3504, 3505, 3511, 3512, 3519, 3523, 3525, 3557, 3562, 3585, 3587, 3588, 3596, 3606, 3610, 3616, 3638, 3643, 3666, 3669, 3680, 3712, 3716, 3720, 3731, 3743, 3753, 3763, 3776, 3782, 3800, 3801, 3810, 3812, 3815, 3816, 3836, 3844, 3850, 3857, 3877, 3879, 3895, 3942, 3951, 3954, 3967, 3969, 3979, 3980, 3982, 3991, 3995, 4004, 4007, 4014, 4015, 4018, 4028, 4031, 4052, 4057, 4062, 4065, 4070, 4073, 4078, 4084, 4094, 4121, 4164, 4176, 4194, 4195, 4203, 4218, 4227, 4231, 4233, 4240, 4245, 4252, 4257, 4258, 4276, 4281, 4334, 4335, 4338, 4339, 4345, 4372, 4375, 4390, 4391, 4406, 4414, 4426, 4432, 4437, 4443, 4444, 4449, 4456, 4479, 4487, 4490, 4491, 4508, 4514, 4522, 4533, 4536, 4575, 4604, 4608, 4615, 4617, 4630, 4655, 4659, 4690, 4704, 4710, 4740, 4747, 4748, 4751, 4762, 4763, 4771, 4781, 4787, 4796, 4799, 4830, 4833, 4838, 4839, 4840, 4856, 4862, 4866, 4867, 4868, 4876, 4879, 4887, 4896, 4900, 4904, 4914, 4916, 4933, 4946, 4959, 4967, 4984, 4989, 4992, 5010, 5015, 5042, 5048, 5050, 5069, 5079, 5082, 5085, 5115, 5120, 5131, 5143, 5153, 5161, 5163, 5167, 5172, 5196, 5220, 5228, 5230, 5243, 5247, 5253, 5254, 5261, 5277, 5295, 5299, 5303, 5325, 5330, 5357, 5370, 5385, 5394, 5415, 5419, 5421, 5435, 5444, 5447, 5468, 5475, 5481, 5485, 5489, 5493, 5498, 5510, 5512, 5518, 5524, 5525, 5540, 5546, 5556, 5568, 5587, 5594, 5602, 5613, 5618, 5620, 5622, 5629, 5631, 5641, 5645, 5651, 5660, 5677, 5698, 5702, 5706, 5715, 5718, 5722, 5727, 5747, 5748, 5749, 5751, 5761, 5781, 5794, 5796, 5797, 5814, 5834, 5841, 5854, 5860, 5880, 5887, 5891, 5915, 5923, 5924, 5931, 5936, 5944, 5947, 5950, 5952, 5959, 5965, 5967, 5978, 6001, 6008, 6013, 6017, 6038, 6043, 6045, 6055, 6057, 6065, 6067, 6068, 6079, 6081, 6082, 6088, 6090, 6095, 6105, 6113, 6130, 6136, 6139, 6145, 6149, 6171, 6174, 6178, 6182, 6217, 6219, 6224, 6229, 6233, 6239, 6246, 6247, 6272, 6282, 6297, 6310, 6358, 6398, 6401, 6410, 6417, 6424, 6428, 6429, 6439, 6441, 6463, 6471, 6485, 6487, 6497, 6510, 6520, 6535, 6551, 6566, 6571, 6576, 6589, 6611, 6626, 6641, 6653, 6655, 6668, 6673, 6685, 6693, 6701, 6722, 6731, 6744, 6761, 6784, 6788, 6803, 6827, 6832, 6876, 6877, 6887, 6888, 6889, 6910, 6915, 6920, 6926, 6939, 6964, 6977, 6994, 6996, 7015, 7017, 7019, 7024, 7028, 7033, 7046, 7048, 7057, 7081, 7083, 7084, 7092, 7102, 7105, 7121, 7125, 7131, 7137, 7138, 7143, 7154, 7162, 7163, 7173, 7180, 7190, 7200, 7211, 7220, 7223, 7224, 7251, 7255, 7259, 7272, 7292, 7293, 7300, 7305, 7306, 7330, 7331, 7336, 7342, 7343, 7356, 7370, 7374, 7393, 7420, 7451, 7452, 7464, 7495, 7503, 7505, 7509, 7512, 7521, 7531, 7544, 7559, 7562, 7574, 7588, 7592, 7598, 7599, 7604, 7606, 7616, 7647, 7662, 7667, 7677, 7701, 7727, 7728, 7747, 7751, 7765, 7772, 7797, 7803, 7832, 7837, 7839, 7849, 7869, 7870, 7886, 7890, 7901, 7912, 7913, 7919, 7943, 7949, 7959, 7970, 7981, 7982, 7990, 7997, 8023, 8065, 8069, 8076, 8077, 8085, 8088, 8094, 8097, 8104, 8107, 8109, 8116, 8124, 8127, 8161, 8175, 8187, 8195, 8198, 8200, 8231, 8232, 8235, 8244, 8247, 8253, 8270, 8273, 8280, 8288, 8290, 8310, 8313, 8315, 8322, 8326, 8333, 8335, 8337, 8366, 8373, 8386, 8397, 8398, 8408, 8457, 8459, 8471, 8483, 8489, 8492, 8518, 8525, 8536, 8540, 8552, 8574, 8575, 8577, 8581, 8594, 8614, 8619, 8643, 8657, 8697, 8706, 8711, 8712, 8725, 8733, 8734, 8743, 8746, 8750, 8761, 8765, 8792, 8796, 8804, 8806, 8845, 8849, 8859, 8872, 8873, 8877, 8883, 8888, 8894, 8896, 8903, 8916, 8919, 8922, 8938, 8944, 8948, 8949, 8957, 8959, 8961, 8985, 9006, 9025, 9027, 9031, 9035, 9052, 9053, 9076, 9093, 9100, 9145, 9149, 9162, 9168, 9172, 9178, 9179, 9192, 9203, 9204, 9211, 9214, 9237, 9253, 9264, 9272, 9276, 9281, 9287, 9288, 9290, 9300, 9302, 9335, 9338, 9349, 9354, 9362, 9368, 9375, 9389, 9394, 9432, 9437, 9438, 9449, 9468, 9485, 9486, 9487, 9488, 9497, 9501, 9502, 9509, 9513, 9522, 9529, 9531, 9533, 9539, 9542, 9550, 9561, 9566, 9567, 9570, 9581, 9584, 9586, 9590, 9600, 9618, 9622, 9633, 9642, 9657, 9668, 9669, 9684, 9690]


from argparse import ArgumentParser
#from mmd import MMD_loss
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime
from channel_attention import ChannelAttention,ChannelAttention2


class Test_VQADataset(Dataset):
    def __init__(self, features_dir, index=None, max_len=500, feat_dim=2944, scale=1):
        super(Test_VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        self.mean_var = np.zeros((len(index), 1472))
        self.std_var = np.zeros((len(index), 1472))
        self.mean_mean = np.zeros((len(index), 1472))
        self.std_mean = np.zeros((len(index), 1472))
        for i in range(len(index)):        
        
            
            features1 = np.load(features_dir + str(index[i]) + '_VGG16_cat_mean_features.npy').squeeze() # [frame_size, 1472,1,1]
            features2 = np.load(features_dir + str(index[i]) + '_VGG16_cat_std_features.npy').squeeze()  # [frame_size, 1472,1,1]
            features = np.concatenate((features1,features2),1) # [frame_size, 2944,1,1]
            mean_var_each  = np.std(features1,0)
            std_var_each  = np.std(features2,0)
            mean_mean_each = np.mean(features1,0)
            std_mean_each = np.mean(features1,0)
            features = features.squeeze() # [frame_size, 2944]
            
            
            
            if features.shape[0] > max_len:
                features = features[:max_len, :]       
            
            self.length[i] = features.shape[0]
            self.features[i, :features.shape[0], :] = features
            self.mean_var[i, :] = mean_var_each
            self.std_var[i, :] = std_var_each
            self.mean_mean[i, :] = mean_mean_each
            self.std_mean[i, :] = std_mean_each            
            
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx],\
                 self.mean_var[idx],self.std_var[idx],self.mean_mean[idx],self.std_mean[idx]
        return sample

class ANN(nn.Module):
    def __init__(self, input_size=2944, reduced_size=256, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input

def PymidPool(q,layer_num,Att):
    """subjectively-inspired temporal pooling"""
    [b,c] = q.shape
    q_pool = q.mean(0,keepdim=True)
    layers = [pow(2,j) for j in range(1,layer_num)]
    
    qT = q.permute(1,0).unsqueeze(0)
    AttT = Att.permute(1,0).unsqueeze(0)
    
    for layer in  layers:
        pool_func = nn.AdaptiveAvgPool1d(layer)
        t_q = pool_func(qT).squeeze(0).permute(1,0)
        t_wt = pool_func(AttT).squeeze(0).permute(1,0)
        temp = t_q/t_wt
        q_pool = torch.cat((q_pool,temp),0)

    return q_pool
    

class GSTVQA(nn.Module):
    def __init__(self, input_size=1472, reduced_size=256, hidden_size=32,att_frams=15,layer_num = 7):

        super(GSTVQA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)
              
        self.CA  = ChannelAttention(2944,  reduction_dim = 320)
        self.q_reg = torch.nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        self.q_mean = torch.nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        self.q_reg.data.fill_(1)
        self.q_mean.data.fill_(0)
        self.q_reg2 = nn.Linear((pow(2,layer_num)-1), 1)
        self.q_att = nn.Sequential(
                     nn.Conv1d (hidden_size,1,att_frams),
                     nn.ReLU(),
                     nn.Conv1d (1,1,att_frams))
        self.att_frams = att_frams
        self.layer_num = layer_num        
        
    def forward(self, input, input_length,mean_var,std_var,mean_mean,std_mean):
        input = self.CA(input,mean_var,std_var,mean_mean,std_mean)
        input = self.ann(input)  # dimension reduction
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        score = torch.zeros_like(input_length, device=outputs.device)  #
        frame_score = torch.zeros_like(input_length, device=outputs.device)
        
        for i in range(input_length.shape[0]):  #
            qi = outputs[i, :np.int(input_length[i].numpy()),:]
            Att = torch.tanh(self.q_att(qi.permute(1,0).unsqueeze(0))).squeeze().repeat(self.hidden_size,1).permute(1,0)          
            q_fet = (Att*(qi[int(self.att_frams-1):-int(self.att_frams-1),:]))
            qi = PymidPool(q_fet,self.layer_num,Att)
            qi = self.q(qi).permute(1,0)
            if i==0:
               Q_eachV = qi
            else:
               Q_eachV = torch.cat((Q_eachV,qi),0)
            frame_input = outputs[i, :np.int(input_length[i].numpy())].mean(0)-self.q_mean
            frame_score[i] = (torch.exp(-(self.q_reg**2)*frame_input*frame_input)).mean()            
 
        score = self.q_reg2(Q_eachV)

        return score
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0      
      


if __name__ == "__main__":
    parser = ArgumentParser(description='"GSTVQA')
    parser.add_argument("--TrainIndex", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 2000)')

    parser.add_argument('--database', default='KSC-1W', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='GSTVQA', type=str,
                        help='model name (default: GSTVQA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0,
                        help='val ratio (default: 0.2)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    args.decay_interval = int(args.epochs/10)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True
    TestIndex = args.TrainIndex
    if TestIndex==1:    
        datainfo_path = "../datas/CVD2014info.mat"   
        model_path = "./models/training-all-data-GSTVQA-cvd14-EXP0-best" 
    if TestIndex==2:
        datainfo_path = "../datas/LIVE-Qualcomminfo.mat"    
        model_path = "./models/training-all-data-GSTVQA-liveq-EXP0-best"
    if TestIndex==3:
        datainfo_path = "../datas/LIVE_Video_Quality_Challenge_585info.mat" 
        model_path = "./models/training-all-data-GSTVQA-livev-EXP0-best"
    if TestIndex==4:
        datainfo_path = "../datas/KoNViD-1kinfo-original.mat" 
        model_path = "./models/training-all-data-GSTVQA-konvid-EXP0-best"
    
    for test_dataset in range(4):
    
            if test_dataset == 0:
                test_index = [i for i in range(234)]
                print(len(test_index))  
                print('-----------cvd-----------')
                feature_path ="../VGG16_mean_std_features/VGG16_cat_features_CVD2014_original_resolution/"
                           
                
            if test_dataset == 1:
                test_index = [i for i in range(208)]
                print(len(test_index)) 
                print('-----------LIVE-Q-----------')
                feature_path ="../VGG16_mean_std_features/VGG16_cat_features_LIVE-Qua_1080P/"
                
                
            
            if test_dataset == 2:
                test_index = [i for i in range(585)]
                print(len(test_index))
                print('-----------LIVE-V-----------')  
                feature_path ="../VGG16_mean_std_features/VGG_cat_features_LIVE_VQC585_originla_resolution/"
                
                
            
            if test_dataset == 3:
                test_index = [i for i in range(1200)]
                print(len(test_index)) 
                print('-----------KoNViD-----------')
                feature_path ="../VGG16_mean_std_features/VGG16_cat_features_KoNViD_original_resolution/"  


               
        
            print('EXP ID: {}'.format(args.exp_id))
            print(args.database)
            print(args.model)
        
            device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
            features_dir = feature_path 
            datainfo = datainfo_path  
            Info = h5py.File(datainfo, 'r')  
            ref_ids = Info['ref_ids'][0, :]  
            max_len = 500
            
            scale = Info['scores'][0, :].max()  
            test_dataset =  Test_VQADataset(features_dir, test_index, max_len, scale=scale)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
            
            save_result_file = 'results/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)
            criterion = nn.L1Loss().to(device)    
            
            model = GSTVQA().to(device)  #
            
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print("done model")
            with torch.no_grad():
                y_pred = np.zeros(len(test_index))
                y_test = np.zeros(len(test_index))
                L = 0
                for i, (features, length, label,mean_var,std_var,mean_mean,std_mean) in enumerate(test_loader):

                    y_test[i] = scale * label.item()

                    features = features.to(device).float()
                    label = label.to(device).float()
                    mean_var = mean_var.to(device).float()
                    std_var = std_var.to(device).float()
                    mean_mean = mean_mean.to(device).float()
                    std_mean = std_mean.to(device).float()
                    
                    outputs = model(features, length.float(),mean_var,std_var,mean_mean,std_mean)
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()

            test_loss = L / (i + 1)
            PLCC = stats.pearsonr(y_pred, y_test)[0]
            SROCC = stats.spearmanr(y_pred, y_test)[0]
            RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
            KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
            print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(test_loss, SROCC, KROCC, PLCC, RMSE))
            
            
    
