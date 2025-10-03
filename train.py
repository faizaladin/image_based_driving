import torch
from torch.utils.data import DataLoader, random_split
from dataloader import CarlaSteeringDataset # Assumes you have this file
from model import Driving # Assumes you have this file
import wandb
from tqdm import tqdm
import numpy as np

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS = 5000
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use a generator for reproducible train/val splits
generator = torch.Generator().manual_seed(42)

wandb.init(project="carla-driving", name="driving-pilotnet-scaled-loss")

def main():
    # --- Dataset and DataLoader ---
    # The dataset is now expected to return scaled steering values
    dataset = CarlaSteeringDataset(root_dir='./', transform=None) 
    
    # Use torch's random_split for a cleaner implementation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Using device: {DEVICE}")
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    # --- Model, Loss, Optimizer ---
    model = Driving().to(DEVICE)
    criterion = torch.nn.MSELoss() # Sticking with MSE, but now on scaled targets
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # WandB: Watch the model for gradients and topology
    wandb.watch(model, criterion, log="all", log_freq=10)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for i, (images, steers) in enumerate(train_bar):
            images = images.to(DEVICE)
            # Make sure steers are correctly shaped: [batch_size, 1]
            steers = steers.view(-1, 1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, steers)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for i, (images, steers) in enumerate(val_bar):
                images = images.to(DEVICE)
                steers = steers.view(-1, 1).to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, steers)
                val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Logging and Model Saving ---
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "epoch": epoch + 1
        })
        
        # Overwrite the latest model file at the end of every epoch
        model_path = 'latest_driving_model.pth'
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path) # Save latest model to W&B
        print(f"✅ Model saved after epoch {epoch+1}")

    print("Training complete. Final model saved as latest_driving_model.pth")

if __name__ == "__main__":
    main()

