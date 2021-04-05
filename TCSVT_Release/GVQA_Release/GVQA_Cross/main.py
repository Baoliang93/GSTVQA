"""Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2019/11/8
#
# tensorboard --logdir=logs --port=6006
# CUDA_VISIBLE_DEVICES=1 python Training_VSFA_all_data_300epoch.py --database=KoNViD-1k --exp_id=0


from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from channel_attention import ChannelAttention,EarlyStopping
from torch.autograd import Variable
import logging




class Logger(object):

    def __init__(self, name='logger', level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        fh = logging.FileHandler('%s.log' % name, 'w')
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
log = Logger(name='weight_loss') 

class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=500, feat_dim=2944, scale=1):
        super(VQADataset, self).__init__()
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



class D_gauss_net(nn.Module):  
    def __init__(self,hidden_layer_size=64):
        super(D_gauss_net, self).__init__()
        self.lin1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.lin3 = nn.Linear(hidden_layer_size, 1)
    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))
 
    
    
class ANN(nn.Module):
    def __init__(self, input_size=2944, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
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


    
def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


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
        varT = self.q_reg**2
        meanT = self.q_mean

        return score,frame_score,frame_input,varT,meanT

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


if __name__ == "__main__":
    parser = ArgumentParser(description='"GSTVQA')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--TrainIndex", type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 2000)')

    parser.add_argument('--database', default='cvd14', type=str,
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
    SourceIndex = args.TrainIndex
    if SourceIndex ==1:
        args.database = 'cvd14'
        features_dir = "../VGG16_mean_std_features/VGG16_cat_features_CVD2014_original_resolution/"
        datainfo = "../datas/CVD2014info.mat"
    if SourceIndex ==2:
        args.database = 'liveq'
        features_dir = "../VGG16_mean_std_features/VGG16_cat_features_LIVE-Qua_1080P/"
        datainfo = "../datas/LIVE-Qualcomminfo.mat"
    if SourceIndex ==3:    
        args.database = 'livev'
        features_dir = "../VGG16_mean_std_features/VGG_cat_features_LIVE_VQC585_originla_resolution/"
        datainfo = "../datas/LIVE_Video_Quality_Challenge_585info.mat"    
    if SourceIndex ==4:       
        args.database = 'konvid'
        features_dir = "../VGG16_mean_std_features/VGG16_cat_features_KoNViD_original_resolution/"
        datainfo = "../datas/KoNViD-1kinfo-original.mat"
        
    print('Source Dataset: ', args.database)
    print(args.model)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')  # index, ref_ids
    index = Info['index']
    index = index[:, args.exp_id % index.shape[1]]  # np.random.permutation(N)
    ref_ids = Info['ref_ids'][0, :]  #
    max_len = int(Info['max_len'][0])
    trainindex = index[0:int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index)))]
    testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
    train_index, val_index, test_index = [], [], []
    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)
    

    scale = Info['scores'][0, :].max()  # label normalization factor
    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)


    
    hidden_layer_size = 32
    model = GSTVQA().to(device)  #

    D_gauss = D_gauss_net(hidden_layer_size) 
    D_gauss = D_gauss.to(device) 
    D_gauss_optim = Adam(D_gauss.parameters(), lr=0.0001,betas = (0.9,0.999)) 
    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/training-all-data-{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)
      
    criterion = nn.L1Loss()  # L1 loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    best_val_criterion = -1  # SROCC min
    early_stopping = EarlyStopping(10, 0.0005,trained_model_file)
    for epoch in range(args.epochs):
        # Train
        model.train()
        L1 = 0
        L2 = 0
        L3 = 0
        LD = 0
        Num = 0

        for i, (features, length, label,mean_var,std_var,mean_mean,std_mean) in enumerate(train_loader):
            features = features.to(device).float()
            mean_var = mean_var.to(device).float()
            std_var = std_var.to(device).float()
            mean_mean = mean_mean.to(device).float()
            std_mean = std_mean.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #  

            D_gauss_optim.zero_grad()  #  
            outputs,frame_out,z1,varT,meanT= model(features, length.float(),mean_var,std_var,mean_mean,std_mean)
                        
            if epoch<20:
               var = varT.data
               var = torch.ones(label.shape[0],var.shape[0])
               X1 = torch.normal(mean=0, std=var).unsqueeze(0)
               for i in range(label.shape[0]-1):
                    X1 = torch.cat((X1,torch.normal(mean=0, std=var).unsqueeze(0)),0)
              
            elif (epoch%20)==0 and i==0:
               var = varT.data
               mean = meanT.data
               print(var)
               print(mean)
               X1 = torch.normal(mean=mean, std=var).unsqueeze(0)
               for i in range(label.shape[0]-1):
                    X1 = torch.cat((X1,torch.normal(mean=mean, std=var).unsqueeze(0)),0)
            else:
               X1 = torch.normal(mean=mean, std=var).unsqueeze(0)
               for i in range(label.shape[0]-1):
                    X1 = torch.cat((X1,torch.normal(mean=mean, std=var).unsqueeze(0)),0)
 
            
            z_real_gauss = Variable(X1.cuda())
            G_loss_gauss = torch.mean((1-D_gauss(z1))**2)
        
            model.zero_grad()                                         
            loss1 = criterion(outputs, label)
            loss2 =  0.05*G_loss_gauss
            loss3 =  0.5*criterion(frame_out, label)
            loss = loss1+loss2+loss3
            loss.backward()
            optimizer.step()
            
            D_loss_gauss = -torch.mean((D_gauss(z_real_gauss))**2) - torch.mean((1-D_gauss(z1.detach()))**2)
            loss_discriminate = 0.05*D_loss_gauss
            loss_discriminate.backward()            
            D_gauss_optim.step()
            
            L1 = L1 + loss1.item()
            L2 = L2 + loss2.item()
            L3 = L3 + loss3.item()
            Num += (1.0*label.shape[0])/args.batch_size
            
            LD = LD + loss_discriminate.item()
        train_l1 = L1 / Num
        train_l2 = L2 / Num
        train_l3 = L3 / Num
#        early_stopping(train_l1+train_l3, model)
#        if early_stopping.early_stop:
#                print("Early stopping, epoch:",epoch)
#                break
      
        d_loss = LD/ (i + 1)
        log.info(("Dloss", d_loss,'Gloss:',train_l2,'MseLoss',train_l1,\
                  'FramLoss',train_l3," epoch:",epoch))
    torch.save(model.state_dict(), trained_model_file+'-best')

       