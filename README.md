Project Documentation: EDSR with Edge-Aware Loss

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


Prepare Your Environment
Ensure you have the required libraries installed:

        pip install tqdm
        pip install torch torchvision
        pip install pillow
        pip install scikit-image
        pip install numpy
        pip install matplotlib

Prepare Datasets:


Organize your text image datasets as follows:
low_res_dir: Directory with low-resolution images (input).
high_res_dir: Directory with high-resolution images (ground truth).
Similarly, prepare validation and test datasets.

Training the Model:


To train the model, run the main.py script. This will automatically start the training process.





Once the model is trained, you can test it on new images using the test() function inside the train.py script.
I only trained 30 epochs, because after 30 I don't see any changes in training.


Input:
![input_epoch_30_sample_1](https://github.com/user-attachments/assets/ad150d41-c3f0-41b8-b1ef-2277856d7c27)
![input_epoch_20_sample_1](https://github.com/user-attachments/assets/50eca069-7b4a-415b-a9bd-b220cb423291)

Output:
![output_epoch_30_sample_1](https://github.com/user-attachments/assets/41b01f9a-8d7d-4ca5-a0ee-7cfb81b7bcae)
![output_epoch_20_sample_1](https://github.com/user-attachments/assets/3d2d9121-c5b8-4de2-94f2-5b4cebc03cc1)


Conclusion
This project combines the power of EDSR with advanced loss functions to produce high-quality super-resolved images, specifically tailored for text enhancement. By incorporating perceptual and edge-aware losses, the model goes beyond simple pixel accuracy to significantly enhance the clarity.


Feel free to adjust the loss weights and training parameters to suit your specific requirements and continue improving the model's performance!


