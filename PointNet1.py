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

class SAModule1(torch.nn.Module):
    def __init__(self, ratio, radius_list, nn):
        super(SAModule1, self).__init__()
        self.ratio = ratio
        self.radius_list = radius_list
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
      idx = fps(pos, batch, ratio=self.ratio)
      #print("x",x.shape,"idx",idx.shape)
      new_points_list = []
      for i, r in enumerate(self.radius_list):
        row, col = radius(pos, pos[idx], r, batch, batch[idx],
                          max_num_neighbors=64)
       # print("row",row.shape,"col",col.shape,"radius:",r)
        edge_index = torch.stack([col, row], dim=0)
      #  print("edge_index", edge_index.shape)
        new_points = self.conv(x, (pos, pos[idx]), edge_index)
       # print("new_points:",new_points.shape)
        new_points_list.append(new_points)
      pos, batch = pos[idx], batch[idx]
     # print("list size",len(new_points_list))
      new_points_concat = torch.cat(new_points_list, dim=1)
      return new_points_concat, pos, batch

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
    def __init__(self,output_dim):
        super(Net, self).__init__()

        self.sa1_module = SAModule1(0.5, [0.2,0.1], MLP([6, 64, 64]))
        self.sa2_module = SAModule1(0.2,[0.35,0.5], MLP([128 + 3, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([512 + 3, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, output_dim)

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
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        norm = x.norm(p=2, dim=1, keepdim=True)
        x_normalized = x.div(norm)
        return x_normalized
