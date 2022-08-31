import torch
import torch.nn as nn

def low_pass(input: torch.Tensor, n_taps=9):
    # n-tap averaging filter
    with torch.no_grad():
        filter = nn.Conv2d(in_channels=3, out_channels=3,
                           kernel_size=(1,n_taps), stride=1, bias=False,
                           padding='same', groups=3, device=input.device)
        filter.weight.data.fill_(1/n_taps)
        #print(filter.weight.data.shape)
        output = filter(input)
    #print(filter.weight.data)
    return output
