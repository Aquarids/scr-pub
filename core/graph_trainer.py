from utils.logger import Logger
from .graph_net import GraphClusterNet

import os
import torch
import torch.nn as nn
import datetime
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

class GraphTrainer:

    def __init__(self, config, logger: Logger, fact_check=False):
        self.logger = logger
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = 100
        self.batch_size = 16

        self.model = self._init_model(
            config.get('cache_model_path'), fact_check
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 0.1),
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=1e-5,
            max_lr=1e-3,
            step_size_up=500,
            cycle_momentum=False
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def _init_model(self, model_path, fact_check=False):

        input_dim = None
        if fact_check:
            input_dim = 13
        else:
            input_dim = 12

        model = GraphClusterNet(input_dim=input_dim).to(self.device)

        if model_path and os.path.exists(model_path):
            self.logger.info(f"Loading model from {model_path}")
            model_filepath = os.path.join(model_path, "model.pt")
            model.load_state_dict(torch.load(model_filepath, map_location=self.device))
        else:
            self.logger.info("Initializing new model")

        model.to(self.device)
        return model

    def save_model(self, cache_dir):
        filename = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        filepath = os.path.join(cache_dir, filename)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        torch.save(self.model.state_dict(), filepath)
        self.logger.info(f"Model saved to {filepath}")


    def train(self, dataset, epochs=100, batch_size=16):
        self.epochs = epochs
        self.batch_size = batch_size

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        best_loss = float('inf')
        progress_bar = tqdm(range(epochs), desc="Training Progress")
        
        for epoch in range(epochs):
            avg_loss = self._train_epoch(dataloader, epoch)
            self.scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.9f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            progress_bar.update()
            
        progress_bar.close()
        self.logger.info("Training completed.")
        return self.model

    def _train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            pred = self.model(batch)
            
            loss = self.criterion(pred, batch.y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=2.0,
                norm_type=2
            )
            self.optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(dataloader.dataset)


    def eval(self, dataset, batch_size=16, save_results=False):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0
        all_preds = []
        all_labels = []
        results = []

        try:
            with torch.no_grad():
                progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
                for batch in progress_bar:
                    output = self.model(batch)
                    pred = torch.sigmoid(output)
                    
                    loss = self.criterion(pred, batch.y)
                    total_loss += loss.item() * batch.num_graphs
                    
                    all_preds.append(pred.cpu())
                    all_labels.append(batch.y.cpu())
                    
                    if save_results:
                        results.extend([{
                            'true_label': batch.y[i].item(),
                            'pred_prob': pred[i].item()
                        } for i in range(len(batch.y))])

                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                
                metrics = {
                    'loss': total_loss / len(dataset),
                    'accuracy': accuracy_score(all_labels.numpy(), (all_preds > 0.5).float().numpy()),
                    'precision': precision_score(all_labels.numpy(), (all_preds > 0.5).float().numpy()),
                    'recall': recall_score(all_labels.numpy(), (all_preds > 0.5).float().numpy()),
                    'f1': f1_score(all_labels.numpy(), (all_preds > 0.5).float().numpy()),
                    'roc_auc': roc_auc_score(all_labels.numpy(), all_preds.numpy()),
                }

                cm = confusion_matrix(all_labels.numpy(), (all_preds > 0.5).float().numpy())
                tn, fp, fn, tp = cm.ravel()

                metrics.update({
                    'confusion_matrix': cm,
                    'TPR': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'TNR': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0
                })

                if save_results:
                    result_dir = os.path.join(self.config.get('output_dir'), 'eval_results')
                    os.makedirs(result_dir, exist_ok=True)
                    filename = f"eval_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    pd.DataFrame(results).to_csv(os.path.join(result_dir, filename), index=False)
                    self.logger.info(f"Results saved to {filename}")

                self.logger.info(
                    f"Evaluation Results - Loss: {metrics['loss']:.4f} | "
                    f"Acc: {metrics['accuracy']:.2%} | AUC: {metrics['roc_auc']:.2%} | "
                    f"TPR: {metrics['TPR']:.2%} | TNR: {metrics['TNR']:.2%} | "
                    f"FPR: {metrics['FPR']:.2%} | FNR: {metrics['FNR']:.2%}"
                )

                
                return metrics

        except Exception as e:
            self.logger.log_exception(e)
            return None
