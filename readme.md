# Combating Knowledge Corruption in Agent Systems: A Byzantine-Tolerant Secure Collaborative RAG Framework

This repository provides the SCR code for the method proposed in:

**Combating Knowledge Corruption in Agent Systems: A Byzantine-Tolerant Secure Collaborative RAG Framework**

## Setup

Create the Conda environment:

```
conda env create -f environment.yml
conda activate scr
```

## Train the model

Prepare your training dataset.h5 as following structure

```text
dataset.h5
├─ metadata (group)
│  ├─ @attrs.n_benign:     int
│  ├─ @attrs.n_malicious:  int
│  └─ @attrs.n_doc_each:   int
├─ iter_0 (group)
│  ├─ round_0 (group)
│  │  ├─ features:   float32 [N, F]
│  │  ├─ edge_index: int32   [2, E]
│  │  └─ labels:     int8    [L]   (1=benign, 0=poisoned)
│  ├─ round_1
│  │  ├─ features
│  │  ├─ edge_index
│  │  └─ labels
│  └─ ...
├─ iter_1
│  ├─ round_0
│  │  ├─ features
│  │  ├─ edge_index
│  │  └─ labels
│  └─ ...
└─ ...
```

Then training model with your dataset file and save model

```python
from core.dynamic_graph_trainer import DynamicGraphTrainer
from core.dynamic_graph_dataset import DynamicGraphDataProcessor

dataset_path = "your_dataset_path.h5"
graph_preprocessor = DynamicGraphDataProcessor(dataset_path)
trainer = DynamicGraphTrainer(config, logger, should_commonsense_check)

train_dataset, eval_dataset = graph_preprocessor.prepare_datasets()
model = trainer.train(train_dataset, epochs=epochs, batch_size=batch_size)
trainer.eval(eval_dataset, save_results=True)

save_model(model, save_path="your_path", file_name="your_model_name.pt")
```

## Run

From the repository root:

```
python test.py
```

Or you can run the method on your own dataset by preparing `clients_data` in the expected format, initializing the SCR Assessor, and calling `assessor.assess(...)`.

Create a list where each element corresponds to one client/source. Each client has:
- `source`: a unique string identifier
- `data`: a list of documents
  - each document contains `doc_id` and `content` (and optionally `category`)

Update the model path in config

```python
config["cache_model_path"] = "your_model_path"
```

Initialize the auxiliary LLM, then build `BasicQueryClient` instances and create the `Assessor`.
```python
from agent.auxiliary import Auxiliary
from core.assessor import Assessor
from core.basic_query_client import BasicQueryClient

auxiliary = Auxiliary(config, logger)

clients = []
# each client is a source
for i in range(n_clients):
    client = BasicQueryClient(
        logger,
        f"dataset_{i}",
        f"client_{i}",
        db_path,
        auxiliary,
    )
    clients.append(client)

assessor = Assessor(logger, auxiliary, clients, config)
```

Run the method on your prepared `clients_data`:

```python
result = assessor.assess(clients_data)
print(result)
```
