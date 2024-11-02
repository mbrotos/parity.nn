import json
import torch
from torch.utils.data import DataLoader
import argparse
import os

from models.mlp import MultiLayerPerceptron
from datapipeline import ParityDataset
from models.rnn import SimpleRNN
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
        "--model", type=str, default="rnn", help="Model to use."
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

def train_loop(model, train_loader, loss_fn, optimizer, device, epochs):
    model.train()
    total_step = len(train_loader)

    print("Training...\n")
    print('-'*60)

    for epoch in range(1, epochs+1):
        for step, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = loss_fn(outputs, labels)

            if isinstance(optimizer, torch.optim.Optimizer):
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif optimizer == "random":
                # Random update on range of -10 to 10
                for param in model.parameters():
                    param.data = (torch.rand(param.shape) * 20 - 10).to(device)
            accuracy = ((outputs > 0.5) == (labels > 0.5)).type(torch.FloatTensor).mean()

            if (step+1) % 250 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}' 
                       .format(epoch, epochs, 
                        step+1, total_step, 
                        loss.item(), accuracy))
                print('-'*60)
                if abs(accuracy - 1.0) < 0.0001:
                    print("EARLY STOPPING")
                    return


def test_loop(model, train_loader, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    train_loss = 0
    train_correct = 0
    total_test = 0
    total_train = 0
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        pred = model(X_test)
        test_loss += loss_fn(pred, y_test).item()
        test_correct += ((pred > 0.5) == (y_test > 0.5)).sum().item()
        total_test += X_test.shape[0] * X_test.shape[1]
    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        pred = model(X_train)
        train_loss += loss_fn(pred, y_train).item()
        train_correct += (pred.round() == y_train).sum().item()
        total_train += X_train.shape[0] * X_train.shape[1]

    test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / total_test
    train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / total_train
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
    train_dataset = ParityDataset(seed=42)
    test_dataset = ParityDataset(seed=43)

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
    elif args.model == "rnn":
        model = SimpleRNN()
    else:
        raise ValueError(f"Model {args.model} not supported.")

    # Initialize the optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
    elif args.optimizer == "random":
        optimizer = "random"
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # Initialize the loss function
    loss_fn = torch.nn.BCELoss()

    # Initialize the device
    device = torch.device(args.device)
    model.to(device)

    log.info("Starting training...")
    train_loop(model, train_loader, loss_fn, optimizer, device, args.epochs)
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
