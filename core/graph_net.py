import torch
from torch_geometric.nn import GATConv, BatchNorm
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.data import Data

class GraphClusterNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, heads=2, clusters=2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_dim)
        )
        
        self.gat1 = GATConv(hidden_dim, hidden_dim//2, heads=heads)
        self.gat1_norm = BatchNorm(hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.gat2_norm = BatchNorm(hidden_dim)
        
        self.cluster_prototypes = nn.Parameter(torch.randn(clusters, hidden_dim))
        self.cluster_attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, data: Data):
        x = self.encoder(data.x)  # [num_nodes, hidden_dim]
        
        identity = x
        x = self.gat1(x, data.edge_index)
        x = self.gat1_norm(x + identity)
        x = F.elu(x)
        
        identity = x
        x = self.gat2(x, data.edge_index)
        x = self.gat2_norm(x + identity)
        x = F.elu(x)  # [num_nodes, hidden_dim]
        
        batch_size = 1
        
        # [batch_size, seq_len, embed_dim]
        x_attn = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        prototypes = self.cluster_prototypes.unsqueeze(0).repeat(batch_size, 1, 1)  # [1, clusters, hidden_dim]

        attn_output, _ = self.cluster_attention(
            query=x_attn,
            key=prototypes,
            value=prototypes
        )  # [1, num_nodes, hidden_dim]
        
        cluster_feat = attn_output.squeeze(0)  # [num_nodes, hidden_dim]
        
        combined = torch.cat([x, cluster_feat], dim=1)  # [num_nodes, hidden_dim*2]
        return self.classifier(combined)