Project Documentation: EDSR with Edge-Aware Loss
Overview
This project aims to enhance image resolution using a deep learning-based Super-Resolution approach, specifically the Enhanced Deep Super-Resolution (EDSR) model. The model is extended with custom loss functions including Perceptual Loss and Edge-Aware Loss to preserve image details and edge structures, improving the perceptual quality of super-resolved images.

Key Components:
EDSR Model: A deep learning model for super-resolution that utilizes a series of residual blocks to upscale low-resolution images to high-resolution outputs.

Loss Functions:

L1 Loss: Measures the pixel-wise difference between the predicted and target images.
Perceptual Loss: Uses VGG19 features to compute the perceptual difference between predicted and target images, emphasizing high-level features.
Edge-Aware Loss: Focuses on preserving edges by penalizing differences in edge structures between predicted and target images.
Training and Validation Pipelines: Implemented to train the model using paired low and high-resolution images, validating the model's performance on unseen data.

Project Structure:

EDSR/
├── config.py              # Configuration file containing all hyperparameters and paths
├── main.py                # Main script to start training or testing
├── models/
│   └── model.py           # EDSR model implementation
├── losses/
│   └── loss.py            # Custom loss functions including Perceptual Loss and Edge-Aware Loss
├── train/
│   └── train.py           # Training script with data loading, model initialization, and training loop
├── utils/
│   └── utils.py           # Utility functions for data loading, checkpoint saving, and image processing
└── checkpoints/           # Directory to save model checkpoints

Configuration (config.py)
The Config class holds all configuration parameters for the project:

Model Parameters:

scale: Scaling factor for upscaling images (e.g., 4x).
num_channels: Number of input channels (e.g., 3 for RGB).
num_res_blocks: Number of residual blocks in the EDSR model.
num_features: Number of feature maps in the convolutional layers.
Image Resolution:

HIGH_RES: Target high-resolution image size.
LOW_RES: Low-resolution image size calculated from HIGH_RES.
Training Parameters:

batch_size: Number of images processed in one training step.
learning_rate: Learning rate for the optimizer.
num_epochs: Total number of epochs for training.
Paths:

high_res_dir, low_res_dir: Directories containing high and low-resolution training images.
val_high_res_dir, val_low_res_dir: Directories for validation images.
test_data_dir: Directory for test images.
results_dir: Directory to save the test results.
checkpoint_dir: Directory to save model checkpoints.
How to Use
1. Prepare Your Environment
Ensure you have the required libraries installed:

pip install tqdm
pip install torch torchvision
pip install pillow
pip install scikit-image
pip install numpy
pip install matplotlib

2. Prepare Datasets
Organize your datasets as follows:

low_res_dir: Directory with low-resolution images (input).
high_res_dir: Directory with high-resolution images (ground truth).
Similarly, prepare validation and test datasets.
3. Training the Model
To train the model, run the main.py script. It will automatically start the training process:

python main.py

4. Testing the Model
Once the model is trained, you can test it on new images using the test() function inside the train.py script.

Key Classes and Functions
1. EDSR (models/model.py)
The main model architecture for super-resolution, using residual blocks to upscale images.

2. Loss Functions (losses/loss.py)
PerceptualLoss: Uses VGG19 features to compare perceptual differences.
EdgeAwareLoss: Computes the edge maps of images and compares them to preserve edge structures.
CombinedLoss: Combines L1, Perceptual, and Edge-Aware losses.
3. Training (train/train.py)
The training script handles data loading, model initialization, loss calculation, backpropagation, and checkpoint saving. It validates the model at each epoch to track performance.

4. Utilities (utils/utils.py)
Contains helper functions:

random_crop(): Crops low and high-resolution images randomly to match input sizes.
CustomDataset: A PyTorch Dataset class for loading image pairs.
load_dataset(): Loads and returns DataLoader objects for training, validation, or testing.
save_checkpoint(): Saves the model state during training.
Important Notes
Ensure that all paths are correctly set in config.py.
Adjust learning_rate, batch_size, and other hyperparameters based on your system’s capability and the dataset size.
During testing, ensure the model checkpoint path is correctly set to load the desired weights.

Conclusion
This project combines the power of EDSR with advanced loss functions to produce high-quality super-resolved images, specifically tailored for text enhancement. By incorporating perceptual and edge-aware losses, the model goes beyond simple pixel accuracy to significantly enhance the clarity, edge sharpness, and legibility of text, making it ideal for real-world applications like document processing, OCR improvement, and digital archiving, where the visual quality of text is paramount.



Feel free to adjust the loss weights and training parameters to suit your specific requirements and continue improving the model's performance!
