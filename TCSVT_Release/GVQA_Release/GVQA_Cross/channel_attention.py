import torch
import torch.nn as nn   
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0,trained_model_file='konvid'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.name = trained_model_file

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
#            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.name+'-best')

class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_dim):
        super(ChannelAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.n_channels_in= int(n_channels_in/2)
        self.reduction_dim = reduction_dim        
        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.reduction_dim),
            nn.ReLU(),
            nn.Linear(self.reduction_dim, self.n_channels_in),         
        ).to(self.device)
        self.normar_func = nn.Sigmoid() 
        
        
        

    def forward(self, x,mean_var,std_var,mean_mean,std_mean):
        
        [b,c,l] = x.shape
        x = x.to(self.device)  
        mean_var = mean_var.to(self.device)  
        xx_std =  x[:,:,1472:] 
        att = self.bottleneck(mean_var)
        att = self.normar_func(att)
    #    print("att",att)
        x_std = torch.mul(xx_std, att.unsqueeze(1).repeat(1,c,1))
        
        return x_std

class ChannelAttention2(nn.Module):
    def __init__(self, n_channels_in, reduction_dim):
        super(ChannelAttention2, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.n_channels_in= int(n_channels_in/2)
        self.get_alpha = nn.Linear(self.n_channels_in, 1)
        self.get_beta = nn.Linear(self.n_channels_in, 1)
        
        

    def forward(self, x,mean_var,std_var,mean_mean,std_mean):
        
      
        x = x.to(self.device)  
        xx_std =  x[:,:,1472:] 
        alpha = self.get_alpha(xx_std).repeat(1,1,1472)
        beta = self.get_beta(xx_std) .repeat(1,1,1472)
        x_std = (xx_std*beta)+alpha
      
        
        return x_std

def main():
    
    
    x = torch.randn(16,300, 2944)   # multi-scale features [B,300, 1472]
    mean_var =  torch.randn(16, 1472) 
    CA= ChannelAttention(x.shape[2], reduction_dim = 320)
    y = CA(x,mean_var,mean_var,mean_var,mean_var)
    print(y.shape)
    


if __name__ == "__main__":
    main()
