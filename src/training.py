import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from torch_geometric.loader import DataLoader
from src.models.gcn import GCN
from src.utils.data_loader import load_dataset
from src.utils.graph_utils import create_graph_features, create_gcn_dataset
import numpy as np
from src.metrics.evaluation import evaluate_model

def train(config):
    train_data, test_data, train_labels, test_labels = load_dataset(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs_train, features_train, edge_attrs_train = create_graph_features(train_data, k=config.k_neighbors)
    graphs_test, features_test, edge_attrs_test = create_graph_features(test_data, k=config.k_neighbors)

    train_dataset = create_gcn_dataset(features_train, graphs_train, edge_attrs_train, torch.tensor(train_labels, dtype=torch.long), device)
    test_dataset = create_gcn_dataset(features_test, graphs_test, edge_attrs_test, torch.tensor(test_labels, dtype=torch.long), device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Number of train graphs: {len(train_dataset)}")
    print(f"Number of test graphs: {len(test_dataset)}")
    print(f"Edges in first train graph: {graphs_train[0].shape[1]}")
    print(f"Feature shape in first train graph: {features_train[0].shape}")


    model = GCN(
        in_features=3,
        hidden_features=config.hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        all_train_preds, all_train_labels = [], []

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)
            loss = criterion(out, data.y)
            train_loss += loss.item()

            preds = torch.argmax(out, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(data.y.cpu().numpy())

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_accuracy = (np.array(all_train_preds) == np.array(all_train_labels)).mean()

        model.eval()
        test_loss = 0
        all_test_preds, all_test_labels = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                test_loss += loss.item()

                preds = torch.argmax(out, dim=1)
                all_test_preds.extend(preds.cpu().numpy())
                all_test_labels.extend(data.y.cpu().numpy())

        test_loss /= len(test_loader)
        test_accuracy = (np.array(all_test_preds) == np.array(all_test_labels)).mean()

        print(f"Epoch {epoch+1}/{config.epochs} "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} "
              f"| Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")


    class_names = config.class_names

    #relabel
    evaluate_model(model, test_loader, class_names, relabel=True, save_dir="evaluation")
    #normal label
    evaluate_model(model, test_loader, class_names, relabel=False, save_dir="evaluation")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config):
    train(config)


if __name__ == "__main__":
    main()