import hydra
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from src.models.multi_pitch_estimator import MultiPitchEstimator
from src.models.dataset import MaestroDataset
from src.utils.metrics import compute_metrics
from src.utils.helper import convert_args_to_overrides


@hydra.main(config_path="../configs", config_name="model_config", version_base=None)
def train(cfg: DictConfig):
    # Initialize wandb and merge Hydra config with sweep parameters
    wandb.init(project="maestro-multi-pitch-estimation",
               config=OmegaConf.to_container(cfg, resolve=True))
    # Convert sweep params to OmegaConf
    sweep_config = OmegaConf.create(wandb.config)
    cfg = OmegaConf.merge(cfg, sweep_config)  # Merge both configs
    print(f"Using config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = MaestroDataset(hdf5_path=cfg.data.hdf5_path, split="train")
    val_dataset = MaestroDataset(
        hdf5_path=cfg.data.hdf5_path, split="validation")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = MultiPitchEstimator(
        kernel1_size=(cfg.kernel1_size_x, cfg.kernel1_size_y),
        out_channels1=cfg.out_channels1,
        max_pool_kernel1=(cfg.max_pool_kernel1_x,
                          cfg.max_pool_kernel1_y),
        kernel2_size=(cfg.kernel2_size_x, cfg.kernel2_size_y),
        out_channels2=cfg.out_channels2,
        max_pool_kernel2=(cfg.max_pool_kernel2_x,
                          cfg.max_pool_kernel2_y),
        lstm1_hidden_size=cfg.lstm1_hidden_state,
        dropout_size=cfg.dropout_size,
        lstm2_hidden_size=cfg.lstm2_hidden_state
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
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
    import sys
    # Convert --arg=value to arg=value format for Hydra
    overrides = convert_args_to_overrides(sys.argv[1:])
    sys.argv = [sys.argv[0]] + overrides
    train()
