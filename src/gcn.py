import  torch
from    torch.nn import Linear
from    torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self,num_features,num_classes,hid_dim=16,rand_seed=1234):
        super().__init__()
        
        torch.manual_seed(rand_seed)
        
        self.conv1 = GCNConv(num_features, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)

        self.classifier = Linear(hid_dim, num_classes)

    def forward(self, x, edge_index):
        
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out,h