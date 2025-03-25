import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from math import log as ln

from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm

# from Mamba import Mamba, ModelArgs

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

class MambaBlock(nn.Module):
    def __init__(self, in_channels, n_layer=1, bidirectional=False):
        super(MambaBlock, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        self.backward_blocks = None
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                        Block(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)
        
        return residual

class HNFBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        
        self.filters = nn.ModuleList([
            Conv1d(input_size, hidden_size//4, 3, dilation=dilation, padding=1*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 5, dilation=dilation, padding=2*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 9, dilation=dilation, padding=4*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 15, dilation=dilation, padding=7*dilation, padding_mode='reflect'),
        ])
        
        self.conv_1 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')
        
        self.norm = nn.InstanceNorm1d(hidden_size//2)
        
        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')
        
    def forward(self, x):
        residual = x
        
        filts = []
        for layer in self.filters:
            filts.append(layer(x))
            
        filts = torch.cat(filts, dim=1)
        
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        
        filts = F.leaky_relu(torch.cat([self.norm(nfilts), filts], dim=1), 0.2)
        
        filts = F.leaky_relu(self.conv_2(filts), 0.2)
        
        return filts + residual
    

class EMGMAMBA(nn.Module):
    def __init__(self, in_channels=64, feats=64, n_layer=1):
        super().__init__()
        
        self.conv = nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'), nn.LeakyReLU(0.2))

        self.hnf_encode = HNFBlock(feats, feats, 1)

        self.mamba = MambaBlock(in_channels=feats, n_layer=1, bidirectional=False)
        
        # args = ModelArgs(d_model=2560, n_layer=64, vocab_size=10000)

        # self.mamba = Mamba(args)
        
        self.n_layer = n_layer
        
        
        
        self.hnf_decode = HNFBlock(feats, feats, 1)
        
        

        self.conv_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')

        # self.apply(partial(_init_weights, n_layer=n_layer))
        
    def forward(self, x):
        # print(x.shape)
        x_residual = None
        x = self.conv(x)
        # print(x.shape)
        x = self.hnf_encode(x)
         
        # print(x.shape)
        # print(x.device)
        # print(x.device.index)
        # x, x_residual = self.mamba(x, x_residual, inference_params = None)
        # x = (x + x_residual) if x_residual is not None else x
        
        # x = self.mamba(x.permute(0,2,1)).permute(0,2,1)                   # (B, C(D), L) -> (B, L, C(D)) -> (B, C(D), L)

        for i in range(self.n_layer):
            x = self.mamba(x.permute(0,2,1)).permute(0,2,1) 

        x = self.hnf_decode(x)
        x = self.conv_out(x)
        return x
       
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EMGMAMBA(in_channels = 64, feats=64).to(device)
    input_size = (1, 10000)
    batch_size = 256
    dtypes = [torch.float32]
    result = summary(model, input_size=input_size)
