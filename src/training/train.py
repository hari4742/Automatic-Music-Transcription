import hydra
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig
from src.models.multi_pitch_estimator import MultiPitchEstimator
from src.models.dataset import MaestroDataset
from src.utils.metrics import compute_metrics


@hydra.main(config_path="../configs", config_name="model_config", version_base=None)
def train(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project="maestro-multi-pitch-estimation", config=dict(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = MaestroDataset(hdf5_path=cfg.data.hdf5_path, split="train")
    val_dataset = MaestroDataset(
        hdf5_path=cfg.data.hdf5_path, split="validation")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = MultiPitchEstimator().to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}"):
            cqt, pianoroll = batch

            cqt = cqt.to(device)
            pianoroll = pianoroll.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(cqt)
            loss = criterion(outputs, pianoroll)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute accuracy
            # Convert logits to binary predictions
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == pianoroll).sum().item()
            train_total += pianoroll.numel()

            train_loss += loss.item()

        # Log training loss and accuracy
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                cqt, pianoroll = batch

                cqt = cqt.to(device)
                pianoroll = pianoroll.to(device)

                outputs = model(cqt)
                loss = criterion(outputs, pianoroll)
                val_loss += loss.item()

                # Compute accuracy
                # Convert logits to binary predictions
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == pianoroll).sum().item()
                val_total += pianoroll.numel()

                all_outputs.append(outputs)
                all_targets.append(pianoroll)

         # Log validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

        # Compute metrics (precision, recall, F1)
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        metrics = compute_metrics(outputs, targets)
        wandb.log(metrics)

        # Save the best model
        if val_loss < wandb.run.summary.get("best_val_loss", float("inf")):
            wandb.run.summary["best_val_loss"] = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # Save the final model
    torch.save(model.state_dict(), "final_model.pth")
    wandb.finish()


if __name__ == "__main__":
    train()
