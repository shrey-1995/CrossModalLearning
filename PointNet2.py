import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.data.data import Data

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        #print("x",x.shape,"idx",idx.shape)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        #print("row",row.shape,"col",col.shape)
        edge_index = torch.stack([col, row], dim=0)
        #print("edge_index", edge_index.shape)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 300)

    def forward(self, data):
        # Convert to torch_geometric.data.Data type
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xyz,color = data["xyz"].to(device), data["color"].to(device)
        #print("xyz", xyz.shape, "color", color.shape)
        batch_size, N, _ = data["xyz"].shape  # (batch_size, num_points, 3)
        pos = xyz.view(batch_size*N, -1)
        batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long).to(device)
        for i in range(batch_size): batch[i] = i
        batch = batch.view(-1)
        data = Data()
        data.pos, data.batch = pos.float(), batch
        data.x = color.view(batch_size*N, -1).float()
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)
