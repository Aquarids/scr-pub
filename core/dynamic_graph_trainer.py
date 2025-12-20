from utils.logger import Logger

import os
import math
import torch
import datetime
from .graph_trainer import GraphTrainer
from .dynamic_graph_net import DynamicGraphClusterNet, DynamicFeatureGate

class DynamicGraphTrainer(GraphTrainer):
    def __init__(self, config, logger: Logger, fact_check=False):
        super().__init__(config, logger, fact_check)

    def _init_model(self, model_path, fact_check=False):
        if fact_check:
            model = DynamicGraphClusterNet(static_dim=13).to(self.device)
        else:
            model = DynamicGraphClusterNet(static_dim=12).to(self.device)
        
        if model_path and os.path.exists(model_path):
            model_file = os.path.join(model_path, f"dynamic_{fact_check}_model.pt")
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            self.logger.info(f"Loaded dynamic model from {model_file}")
            
        return model

    def _train_epoch(self, dataloader, epoch): 
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            
            output_dict = self.model.forward_with_topo(batch)
            
            cls_loss = self.criterion(output_dict["pred"], batch.y)
            topo_loss = self.model.compute_topology_loss(
                output_dict["edge_index"],
                output_dict["features"],
                output_dict["cluster_assign"]
            )
            gate_reg = self._gate_regularization()
            
            topo_lambda = 0.5 * (1 + math.cos(epoch / self.epochs * math.pi))
            total_loss = cls_loss + topo_lambda * topo_loss + gate_reg
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += total_loss.item() * batch.num_graphs
        return total_loss / len(dataloader.dataset)

    def _gate_regularization(self):
        # reg
        reg_loss = 0.0
        alpha = self.config.get('gate_reg_alpha', 0.01)
        
        for module in self.model.modules():
            if isinstance(module, DynamicFeatureGate):
                reg_loss += alpha * torch.mean(module.gate_net[0].weight.abs())
                
        return reg_loss

    def save_model(self, cache_dir):
        filename = f"dynamic_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        filepath = os.path.join(cache_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        self.logger.info(f"Dynamic model saved to {filepath}")

    def eval(self, dataset, batch_size=16, save_results=False):
        return super().eval(dataset, batch_size, save_results)
