import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GENConv, \
    BatchNorm, LayerNorm, PNAConv
from torch.nn import Linear, Dropout


class GCN_8_plus(torch.nn.Module):
    def __init__(self, num_features, num_classes, initdim=16, inithead=16, edge_dim=12):
        super(GCN_8_plus, self).__init__()

        self.conv1 = GATConv(num_features, initdim, heads=inithead, edge_dim=edge_dim)
        self.BatchNorm1 = BatchNorm(initdim * inithead)
        self.conv_linear1 = torch.nn.Linear(initdim * inithead, initdim)
        self.BatchNorml1 = BatchNorm(initdim)

        self.conv2 = GATConv(initdim, initdim * 2, heads=int(inithead / 2), edge_dim=edge_dim)
        self.BatchNorm2 = BatchNorm(initdim * inithead)
        self.conv_linear2 = torch.nn.Linear(initdim * inithead, initdim * 2)
        self.BatchNorml2 = BatchNorm(initdim * 2)

        self.conv3 = GATConv(initdim * 2, initdim * 4, heads=int(inithead / 4), edge_dim=edge_dim)
        self.BatchNorm3 = BatchNorm(initdim * inithead)

        # self.drop = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(initdim * inithead, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # block 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv_linear1(x)
        x = self.BatchNorml1(x)
        x = F.relu(x)
        # block2

        x = self.conv2(x, edge_index, edge_attr)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv_linear2(x)
        x = self.BatchNorml2(x)
        x = F.relu(x)

        # block 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.BatchNorm3(x)
        x = F.relu(x)

        # x = self.drop(x)
        x = self.linear(x)

        return x





class PNA_GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, edge_dim=13, hidden_dim=64, dropout_p=0.3, deg=None):
        super(PNA_GNN, self).__init__()

        # Aggregators and scalers from PNA paper
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        towers = 1  # ✅ 确保 num_features % towers == 0
        self.conv1 = PNAConv(in_channels=num_features, out_channels=hidden_dim,
                             aggregators=aggregators, scalers=scalers,
                             edge_dim=edge_dim, towers=towers,
                             pre_layers=1, post_layers=1, divide_input=True,
                             deg=deg)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = PNAConv(in_channels=hidden_dim, out_channels=hidden_dim * 2,
                             aggregators=aggregators, scalers=scalers,
                             edge_dim=edge_dim, towers=towers,
                             pre_layers=1, post_layers=1, divide_input=True,
                             deg=deg)
        self.bn2 = BatchNorm(hidden_dim * 2)

        self.conv3 = PNAConv(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4,
                             aggregators=aggregators, scalers=scalers,
                             edge_dim=edge_dim, towers=towers,
                             pre_layers=1, post_layers=1, divide_input=True,
                             deg=deg)
        self.bn3 = BatchNorm(hidden_dim * 4)

        self.dropout = Dropout(p=dropout_p)
        self.linear = Linear(hidden_dim * 4, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.linear(x)



class GCN_8_plus2(torch.nn.Module):
    def __init__(self, num_features, num_classes, initdim=64, inithead=8, edge_dim=12):
        super(GCN_8_plus2, self).__init__()
        # 边特征非线性变换
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 4),
            nn.ReLU(),
            nn.Linear(edge_dim * 4, edge_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_dim * 2, edge_dim)
        )

        # 第一层：num_features → initdim
        self.conv1 = GATConv(num_features, initdim, heads=inithead, edge_dim=edge_dim)   # 25->512(64*8)
        self.bn1 = BatchNorm(initdim * inithead)  # 512(64*8)
        self.linear1 = nn.Linear(initdim * inithead, initdim)  # 512(64*8)->64
        self.ln1 = LayerNorm(initdim)  # 64
        self.drop1 = nn.Dropout(0.3)
        self.residual_proj1 = nn.Linear(num_features, initdim)  # 残差投影：num_features → initdim    25->64

        # 第二层：initdim → initdim * 2
        self.conv2 = GATConv(initdim, initdim * 2, heads=inithead, edge_dim=edge_dim)  # 64->1024(128*8)
        self.bn2 = BatchNorm(initdim * 2 * inithead)  # 1024(128*8)
        self.linear2 = nn.Linear(initdim * 2 * inithead, initdim * 2)  # 1024(128*8)->128
        self.ln2 = LayerNorm(initdim * 2)  # 128
        self.drop2 = nn.Dropout(0.3)
        self.residual_proj2 = nn.Linear(initdim, initdim * 2)  # 残差投影：initdim → initdim * 2    64->128

        # 第四层：initdim * 2 → initdim * 4
        self.conv3 = GATConv(initdim * 2, initdim * 4, heads=inithead, edge_dim=edge_dim)  # 128->2048(256*8)
        self.bn3 = BatchNorm(initdim * 4 * inithead)  # 2048(256*8)
        self.linear3 = nn.Linear(initdim * 4 * inithead, initdim * 4)  # 2048(256*8)->256
        self.ln3 = LayerNorm(initdim * 4) # 256
        self.drop3 = nn.Dropout(0.3)
        self.residual_proj3 = nn.Linear(initdim * 2, initdim * 4)  # 残差投影：initdim * 2 → initdim * 4  # 128->256

        # 输出层
        self.linear = nn.Linear(initdim * 4, num_classes)  # 128(164*2) -> num_classes

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 边特征处理
        edge_attr = self.edge_mlp(edge_attr)

        # 第一层
        x_in1 = x  # 保存输入用于残差   # 25
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)  # 64
        x = x + self.residual_proj1(x_in1)
        x = self.drop1(x)  # 64

        # 第二层 + 残差
        x_in2 = x  # 保存输入用于残差   64
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)  # 128
        x = x + self.residual_proj2(x_in2)
        x = self.drop2(x)    # 128

        # 第三层
        x_in3 = x   # 128
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.drop3(x)   # 256
        x = x + self.residual_proj3(x_in3)
        x = self.drop3(x)

        # 节点级输出
        x = self.linear(x)
        return x




class GCN_Layer_4(torch.nn.Module):
    def __init__(self, num_features, num_classes, initdim=16, inithead=16):
        super(GCN_Layer_4, self).__init__()
        self.conv1 = GATConv(num_features, initdim, heads=inithead, edge_dim=3)
        self.BatchNorm1 = BatchNorm(initdim * inithead)
        self.conv_linear1 = torch.nn.Linear(initdim * inithead, initdim)
        self.BatchNorml1 = BatchNorm(initdim)

        self.conv2 = GATConv(initdim, initdim * 2, heads=int(inithead / 2), edge_dim=3)
        self.BatchNorm2 = BatchNorm(initdim * inithead)
        self.conv_linear2 = torch.nn.Linear(initdim * inithead, initdim * 2)
        self.BatchNorml2 = BatchNorm(initdim * 2)

        self.conv3 = GATConv(initdim * 2, initdim * 4, heads=int(inithead / 4), edge_dim=3)
        self.BatchNorm3 = BatchNorm(initdim * inithead)
        self.conv_linear3 = torch.nn.Linear(initdim * inithead, initdim * 4)
        self.BatchNorml3 = BatchNorm(initdim * 4)

        self.conv4 = GATConv(initdim * 4, initdim * 8, heads=int(inithead / 8), edge_dim=3)
        self.BatchNorm4 = BatchNorm(initdim * inithead)

        # self.drop = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(initdim * inithead, num_classes)

    def forward(self, data):
        x = data.x

        adj = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        # print("batch",data.batch.shape)
        # edge_attr = data.edge_attr
        # x, att1 = self.conv1(x, adj, return_attention_weights=True)

        # block 1
        x = self.conv1(x, adj, edge_attr)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv_linear1(x)
        x = self.BatchNorml1(x)
        x = F.relu(x)

        # block2
        x = self.conv2(x, adj, edge_attr)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv_linear2(x)
        x = self.BatchNorml2(x)
        x = F.relu(x)

        # block 3
        x = self.conv3(x, adj, edge_attr)
        x = self.BatchNorm3(x)
        x = F.relu(x)
        x = self.conv_linear3(x)
        x = self.BatchNorml3(x)
        x = F.relu(x)

        # block 4
        x = self.conv4(x, adj, edge_attr)
        x = self.BatchNorm4(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        # x = self.drop(x)
        x = self.linear(x)

        return x


class GCN_block(torch.nn.Module):
    def __init__(self, input_dims, output_dims, head_nums, do_linear=True, linear_outdims=None):
        super(GCN_block, self).__init__()

        self.do_linear = do_linear
        self.conv0 = GATConv(input_dims, output_dims, heads=head_nums, edge_dim=3)
        self.BN0 = BatchNorm(output_dims * head_nums)
        self.relu = torch.nn.ReLU()
        if self.do_linear:
            self.linear = torch.nn.Linear(output_dims * head_nums, linear_outdims)
            self.BN1 = BatchNorm(linear_outdims)

    def forward(self, x, adj, edge_attr):

        x = self.conv0(x, adj, edge_attr=edge_attr)
        x = self.BN0(x)
        x = self.relu(x)

        if self.do_linear:
            x = self.linear(x)

            x = self.BN1(x)
            x = self.relu(x)

        return x


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, init_out_dim=16, init_head_num=48):
        super(GCN, self).__init__()

        self.block1 = GCN_block(num_features, init_out_dim, init_head_num, linear_outdims=init_out_dim)  # 10 ->16

        self.block2 = GCN_block(init_out_dim, init_out_dim * 2, int(init_head_num / 2),
                                linear_outdims=init_out_dim * 2)  # 16 ->32

        self.block3 = GCN_block(init_out_dim * 2, init_out_dim * 4, int(init_head_num / 4),
                                linear_outdims=init_out_dim * 4)  # 32 ->64

        self.block4 = GCN_block(init_out_dim * 4, init_out_dim * 8, int(init_head_num / 8),
                                linear_outdims=init_out_dim * 8)  # 64 -> 128

        self.block5 = GCN_block(init_out_dim * 8, init_out_dim * 16, int(init_head_num / 16),
                                do_linear=False)  # 128 -> 256

        self.head = torch.nn.Linear(init_out_dim * init_head_num, num_classes)

    def forward(self, data):
        x = data.x

        adj = data.edge_index
        edge_attr = data.edge_attr

        batch = data.batch

        x = self.block1(x, adj, edge_attr)
        x = self.block2(x, adj, edge_attr)
        x = self.block3(x, adj, edge_attr)
        x = self.block4(x, adj, edge_attr)
        x = self.block5(x, adj, edge_attr)

        x = global_mean_pool(x, batch)
        x = self.head(x)
        return x



class GAT_3(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT_3, self).__init__()
        self.conv1 = GATConv(num_features, 16, heads=20)
        self.BatchNorm1 = BatchNorm(16 * 20)
        self.conv_linear1 = torch.nn.Linear(16 * 20, 16)
        self.BatchNorml1 = BatchNorm(16)

        self.conv2 = GATConv(16, 32, heads=16)
        self.BatchNorm2 = BatchNorm(32 * 16)
        self.conv_linear2 = torch.nn.Linear(32 * 16, 32)
        self.BatchNorml2 = BatchNorm(32)

        self.conv3 = GATConv(32, 48, heads=8)
        self.BatchNorm3 = BatchNorm(48 * 8)

        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(48*8, num_classes)

    def forward(self, data):
        x = data.x

        adj = data.edge_index
        batch = data.batch

        # block 1
        x = self.conv1(x, adj)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv_linear1(x)
        x = self.BatchNorml1(x)
        x = F.relu(x)

        # block2
        x = self.conv2(x, adj)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv_linear2(x)
        x = self.BatchNorml2(x)
        x = F.relu(x)

        # block 3
        x = self.conv3(x, adj)
        x = self.BatchNorm3(x)

        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.drop(x)
        x = self.linear(x)

        return x



if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(0)

    # 初始化模型
    num_features = 25  # 节点特征维度
    num_classes = 12  # 类别数
    edge_dim = 12  # 边特征维度
    model = GCN_8_plus2(num_features=num_features, num_classes=num_classes, initdim=64, inithead=8, edge_dim=edge_dim)

    # 创建随机图数据
    num_nodes = 100  # 模拟100个粒球
    x = torch.randn(num_nodes, num_features)  # 随机节点特征 [100, 25]
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # 随机边 [2, 200]
    edge_attr = torch.randn(num_nodes * 2, edge_dim)  # 随机边特征 [200, 12]
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 运行前向传播
    output = model(data)

    # 打印输出形状
    print(f"输出形状: {output.shape}")  # 预期: [100, 12]


