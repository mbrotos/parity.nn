import json
import torch
from torch.utils.data import DataLoader
import argparse
import os

from models.mlp import MultiLayerPerceptron
from datapipeline import ParityDataset
from utils import setup_logger, log_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network model."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="train_exp",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--model", type=str, default="mlp", help="Model to use."
    )
    parser.add_argument(
        "--dataset", type=str, default="./data", help="Path to the dataset."
    )
    parser.add_argument(
        "--bitstring_length",
        type=int,
        default=10,
        help="Length of the bit strings.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamw", help="Optimizer to use."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training.",
    )
    return parser.parse_args()


def train_loop(model, train_loader, loss_fn, optimizer, device):
    model.train()
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(model, train_loader, test_loader, loss_fn, device):
    model.eval()
    train_loss = 0
    test_loss = 0
    train_correct = 0
    test_correct = 0
    with torch.no_grad():
        for test_batch, train_batch in zip(test_loader, train_loader):
            X_test, y_test = test_batch
            X_train, y_train = train_batch
            X_test, y_test = X_test.to(device), y_test.to(device)
            X_train, y_train = X_train.to(device), y_train.to(device)
            pred = model(X_test)
            test_loss += loss_fn(pred, y_test).item()
            test_correct += (pred.round() == y_test).sum().item()
            pred = model(X_train)
            train_loss += loss_fn(pred, y_train).item()
            train_correct += (pred.round() == y_train).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = test_correct / len(test_loader.dataset)
    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)

    log.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    log.info(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
    }


def main(args, log):
    log.info(f"Starting training with the following arguments: {args}")

    log.info("Loading the dataset...")
    train_dataset = ParityDataset(
        os.path.join(args.dataset, "train.csv"), args.bitstring_length
    )
    test_dataset = ParityDataset(
        os.path.join(args.dataset, "test.csv"), args.bitstring_length
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    log.info("Initializing the model...")
    if args.model == "mlp":
        model = MultiLayerPerceptron(args.bitstring_length)
    else:
        raise ValueError(f"Model {args.model} not supported.")

    # Initialize the optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # Initialize the loss function
    loss_fn = torch.nn.BCELoss()

    # Initialize the device
    device = torch.device(args.device)
    model.to(device)

    log.info("Starting training...")
    for epoch in range(args.epochs):
        log.info(f"Epoch {epoch+1}")
        train_loop(model, train_loader, loss_fn, optimizer, device)
        metrics = test_loop(model, train_loader, test_loader, loss_fn, device)

    log.info("Training complete.")

    log.info("Saving the model...")
    results_dir = os.path.join("results", args.exp_name)
    os.makedirs(results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
    log.info("Model saved.")

    log.info("Saving metrics...")
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info("Metrics saved.")


if __name__ == "__main__":
    args = parse_args()
    log = setup_logger(log_dir(args.exp_name))
    main(args, log)
