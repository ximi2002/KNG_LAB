from .graph_dataset import (
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)

GRAPH_CLASSIFICATION_DSETS = ["collab", "imdb-binary", "imdb-multi", "rdt-b", "rdt-5k"]
LINK_PREDICTION_DSETS=["collab","ddi"]

__all__ = [
    "GRAPH_CLASSIFICATION_DSETS",
    "LoadBalanceGraphDataset",
    "GraphClassificationDataset",
    "GraphClassificationDatasetLabeled",
    "NodeClassificationDataset",
    "NodeClassificationDatasetLabeled",
    "worker_init_fn",
]
