from torch.nn import Linear, GRU
from torch.nn.functional import relu

_EPSILON = 1e-08

##### USER-DEFINED FUNCTIONS
def log(x):
    return torch.log(x + _EPSILON)

def div(x, y):
    return torch.div(x, (y + _EPSILON))

class sharedSubnetwork(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, x):
         
         return None

