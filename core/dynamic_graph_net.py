from .graph_net import GraphClusterNet

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_mean

class DynamicFeatureGate(nn.Module):
    def __init__(self, input_dim=1, gate_dim=8):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim*2, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, 1),
            nn.Sigmoid()
        )
        self.feature_modulator = nn.Linear(1, gate_dim)
        
    def forward(self, current, history):
        delta = torch.abs(current - history)
        
        gate_input = torch.cat([current, delta], dim=1)
        gate_value = self.gate_net(gate_input)
        
        modulated_feat = self.feature_modulator(current) * gate_value
        
        return gate_value, modulated_feat

class DynamicGraphClusterNet(GraphClusterNet):

    def __init__(self, static_dim, edge_threshold=0.3, hidden_dim=64, gate_dim=2, heads=2, clusters=2):
        self.community_temp = 0.5
        self.edge_threshold = edge_threshold
        dynamic_dim = static_dim + gate_dim + 1

        super().__init__(dynamic_dim, hidden_dim, heads, clusters)
        self.feature_gate = DynamicFeatureGate(input_dim=1, gate_dim=gate_dim)
        self.reliability_proj = nn.Sequential(
            nn.Linear(1, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )

        self.register_buffer('hist_reliability', None)

    def forward(self, data: Data):
        return self.forward_with_topo(data)["pred"]
    
    def forward_with_topo(self, data: Data):

        reliability_stack = data.x[:, -2:]
        history_reliability = reliability_stack[:, 0]
        current_reliability = reliability_stack[:, 1]

        static_stack = data.x[:, :-2]
        data.edge_index = self.dynamic_update_edges(data.edge_index, current_reliability)
        dynamic_features = self.build_dynamic_features(static_stack, history_reliability, current_reliability)

        x = self.encoder(dynamic_features)
        
        identity = x.clone()
        x = self.gat1(x, data.edge_index)
        x = self.gat1_norm(x + identity)
        gat1_out = F.elu(x)
        
        identity = gat1_out.clone()
        x = self.gat2(gat1_out, data.edge_index)
        x = self.gat2_norm(x + identity)
        gat2_out = F.elu(x)

        x_attn = gat2_out.unsqueeze(0)
        prototypes = self.cluster_prototypes.unsqueeze(0).repeat(1, 1, 1)
        attn_output, attn_weights = self.cluster_attention(
            query=x_attn,
            key=prototypes,
            value=prototypes
        )
        
        cluster_assign = torch.argmax(attn_weights.squeeze(0), dim=1)  # [num_nodes]
        combined = torch.cat([gat2_out, attn_output.squeeze(0)], dim=1)
        pred = self.classifier(combined)

        return {
            "pred": pred,
            "edge_index": data.edge_index,
            "features": gat2_out,
            "cluster_assign": cluster_assign
        }

    def dynamic_update_edges(self, edge_index, reliability):
        src, dst = edge_index
        diff = torch.abs(reliability[src] - reliability[dst])
        mask = (diff <= self.edge_threshold)
        return edge_index[:, mask]

    def build_dynamic_features(self, x, hist_reliability, current_reliability):
        static_feat = x
        
        current = current_reliability.unsqueeze(1)
        history = hist_reliability.unsqueeze(1)
        
        gate_value, gated_feat = self.feature_gate(current, history)
        rel_feat = self.reliability_proj(current)
        
        return torch.cat([static_feat, gated_feat, rel_feat], dim=1)

    def compute_topology_loss(self, edge_index, features, cluster_assign):
        edge_stable_loss = self._edge_stability_loss(edge_index, features)
        community_loss = self._community_consistency_loss(cluster_assign, features)
        
        return edge_stable_loss + community_loss

    def _edge_stability_loss(self, edge_index, features):
        src, dst = edge_index
        sim_matrix = torch.cosine_similarity(features[src], features[dst])
        return torch.var(sim_matrix)

    def _community_consistency_loss(self, cluster_assign, features):
        centroids = scatter_mean(features, cluster_assign, dim=0)  
        
        intra_sim = torch.cosine_similarity(features, centroids[cluster_assign])
        inter_sim = torch.logsumexp(
            torch.cdist(features, centroids) / self.community_temp, 
            dim=1
        )
        return -torch.mean(intra_sim - inter_sim)