import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from utils.utils import load_dataset, save_checkpoint
from config import Config
from models.model import EDSR
from losses.loss import CombinedLoss
import os
from torchvision.utils import save_image
import torch.nn as nn

# Function to initialize model weights using Xavier normal initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Main training function
def train():
    # Load training and validation datasets
    start_time = time.time()
    train_loader = load_dataset(
        lowres_dir=Config.low_res_dir,
        highres_dir=Config.high_res_dir,
        hr_crop_size=Config.HIGH_RES,
        scale=Config.scale,
        batch_size=Config.batch_size,
        shuffle=True
    )

    val_loader = load_dataset(
        lowres_dir=Config.val_low_res_dir,
        highres_dir=Config.val_high_res_dir,
        hr_crop_size=Config.HIGH_RES,
        scale=Config.scale,
        batch_size=Config.batch_size,
        shuffle=False
    )
    end_time = time.time()
    print(f"Data loading took {end_time - start_time:.2f} seconds.")

    # Initialize model
    model = EDSR(Config.scale, Config.num_channels, Config.num_res_blocks)
    model.apply(init_weights)

    # Load pretrained weights if specified
    if Config.use_pretrained and Config.mode == 'train':
        try:
            checkpoint = torch.load(Config.weights_path)
            model.load_state_dict(checkpoint['model_state_dict']) 
            print("Loaded pretrained weights.")
        except FileNotFoundError:
            print("Pretrained weights not found, starting training from scratch.")
        except KeyError as e:
            print(f"Error loading weights: {e}")

    # Move model to device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=Config.learning_rate)

    # Training loop
    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0

        # Iterate through training batches
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Config.num_epochs}]", leave=True)

        for i, batch in enumerate(loop):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Loss calculations
            l1_loss_val = criterion.l1_loss(outputs, targets)
            perceptual_loss_val = criterion.perceptual_loss(outputs, targets)
            edge_loss_val = criterion.edge_aware_loss(outputs, targets)
            total_loss = l1_loss_val + perceptual_loss_val + edge_loss_val

            # Backpropagation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += total_loss.item()

            # Update progress bar
            loop.set_postfix(total_loss=running_loss / (i + 1))

        # Validation loss calculation
        val_loss = 0.0
        val_steps = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_steps += 1

        val_loss /= val_steps
        print(f"Validation Loss: {val_loss:.4f}")

        # Save samples and checkpoint
        if epoch % Config.SAVE_SAMPLE_EVERY_N_EPOCH == 0:
            if Config.SAVE_HIGH_RES_LOW_RES:
                save_sample_images(model, val_loader, epoch, device)

        save_checkpoint(model, optimizer, Config.checkpoint_dir, epoch)

        # Save model weights at the final epoch
        if epoch == Config.num_epochs - 1:
            torch.save(model.state_dict(), Config.weights_path)
            print(f"Model weights saved at {Config.weights_path}")

# Function to save sample images during training
def save_sample_images(model, val_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            outputs = torch.clamp(outputs, 0, 1)
            input_save_path = os.path.join(Config.results_dir, f"input_epoch_{epoch+1}_sample_{idx+1}.png")
            output_save_path = os.path.join(Config.results_dir, f"output_epoch_{epoch+1}_sample_{idx+1}.png")

            # Save input and output images
            save_image(inputs[0].cpu(), input_save_path)
            save_image(outputs[0].cpu().detach(), output_save_path)
            print(f"Saved input at {input_save_path} and output at {output_save_path}")

            if idx == 0:  # Save only the first sample to avoid clutter
                break

# Function to test the model on the test dataset
def test():
    # Initialize model
    model = EDSR(Config.scale, Config.num_channels, Config.num_res_blocks)

    # Load pretrained weights
    try:
        checkpoint = torch.load(Config.weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pretrained weights for testing.")
    except FileNotFoundError:
        print("Weights file not found, please train the model first.")
        return
    except KeyError as e:
        print(f"Error loading weights: {e}")
        return

    # Move model to device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_loader = DataLoader(
        load_dataset(Config.test_data_dir, Config.test_data_dir, hr_crop_size=Config.HIGH_RES, scale=Config.scale, is_test=True),
        batch_size=1, shuffle=False, num_workers=0  # Set num_workers to 0 for simplicity during testing
    )

    # Create results directory if it doesn't exist
    os.makedirs(Config.results_dir, exist_ok=True)
    
    # Iterate over the test images
    for idx, inputs in enumerate(test_loader):
        if isinstance(inputs, list):
            inputs = inputs[0]  # Get the first element (lowres_image)

        # Move inputs to device
        inputs = inputs.to(device)

        # Model upscaling
        outputs = model(inputs)

        # Ensure the output is clamped to the range [0, 1]
        outputs = torch.clamp(outputs, 0, 1)

        # Save the upscaled image
        output_save_path = os.path.join(Config.results_dir, f"output_{idx + 1}.png")
        save_image(outputs.squeeze(0).cpu(), output_save_path)
        print(f"Saved upscaled image at {output_save_path}")
