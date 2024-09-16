class Config:
    # Model parameters
    scale = 4  # Scaling factor for super-resolution
    num_channels = 3  # Number of input channels (3 for RGB images)
    num_res_blocks = 24  # Number of residual blocks in the EDSR model
    num_features = 64  # Number of features in each layer of the model

    # Image resolution settings
    HIGH_RES = 180  # High-resolution image size
    LOW_RES = HIGH_RES // 4  # Low-resolution image size based on the scaling factor
    SAVE_HIGH_RES_LOW_RES = True  # Save a sample of high-res and low-res images if True
    SAVE_SAMPLE_EVERY_N_EPOCH = 1  # Save samples every N epochs

    # Training parameters
    batch_size = 16  # Number of samples per batch
    learning_rate = 1e-3  # Learning rate for the optimizer
    num_epochs = 301  # Total number of epochs for training

    # Mode settings
    mode = 'train'  # Set to 'train' for training, 'test' for testing
    use_pretrained = False  # Use pretrained weights during training if True
    weights_path =  # Path to save or load model weights

    # Dataset directories
    high_res_dir =  # Directory for high-resolution training images
    low_res_dir =  # Directory for low-resolution training images

    val_high_res_dir =  # Directory for high-res validation images
    val_low_res_dir =  # Directory for low-res validation images

    test_data_dir =  # Directory for test images

    # Results saving directory
    results_dir =  # Directory to save test results

    # Checkpoint and logging
    checkpoint_dir =  # Directory to save model checkpoints
    log_dir =  # Directory for saving logs

config = Config()
