"""
Pixel Coordinate Prediction using Lightweight CNN
Author: ML Assignment - Supervised Regression
Description: Predict (x,y) coordinates of a white pixel in 50x50 grayscale images

Dataset Design:
- 7,500 samples (3× coverage of 2,500 possible positions)
- Uniform random distribution
- Normalization: images [0,1], coordinates [0,1] via x/49, y/49

Model: Lightweight CNN optimized for single-pixel detection
- Conv2D(8) → MaxPool → Conv2D(16) → MaxPool → Dense(32) → Output(2)
- Total parameters: ~75,000

Metrics: MSE (primary), MAE, Euclidean Distance
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def generate_dataset(num_samples: int, 
                     image_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset of images with a single white pixel at random positions.
    
    Rationale:
    - Total positions: 50×50 = 2,500
    - Dataset size: 7,500 = 3× coverage
    - Ensures uniform learning without excessive redundancy
    
    Args:
        num_samples: Number of images to generate
        image_size: Size of square image (default: 50)
    
    Returns:
        Tuple of (images, coordinates)
    """
    images = np.zeros((num_samples, image_size, image_size), dtype=np.float32)
    coordinates = np.zeros((num_samples, 2), dtype=np.float32)
    
    for i in range(num_samples):
        # Generate random coordinates
        x = np.random.randint(0, image_size)
        y = np.random.randint(0, image_size)
        
        # Set pixel to 1 (normalized from 255)
        images[i, y, x] = 1.0
        
        # Store normalized coordinates [0, 1] using x/49, y/49
        coordinates[i] = [x / (image_size - 1), y / (image_size - 1)]
    
    # Reshape to add channel dimension
    images = images.reshape(num_samples, image_size, image_size, 1)
    
    return images, coordinates


def build_lightweight_cnn(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Build lightweight CNN model optimized for single pixel detection.
    
    Architecture:
        Input (50, 50, 1)
        Conv2D(8, 3x3, padding='same') + ReLU
        MaxPooling(2x2) → (25, 25, 8)
        Conv2D(16, 3x3) + ReLU
        MaxPooling(2x2) → (12, 12, 16)
        Flatten → (2304)
        Dense(32) + ReLU
        Dense(2) → (x, y) coordinates
    
    Design rationale:
    - Lightweight: Task is simple, avoids overfitting
    - Two conv blocks: Extract features at different scales
    - Small filter counts (8, 16): Sufficient for detection
    - No dropout/BatchNorm: Clean dataset doesn't require heavy regularization
    
    Args:
        input_shape: Shape of input images (height, width, channels)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        
        # Output layer - 2 neurons for (x, y) coordinates
        layers.Dense(2)
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model


def evaluate_model(model: keras.Model, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray,
                   image_size: int = 50) -> None:
    """
    Evaluate model with comprehensive metrics.
    
    Metrics:
    1. MSE: Mean Squared Error (primary, matches loss)
    2. MAE: Mean Absolute Error (average pixel error)
    3. Euclidean Distance: Direct geometric distance
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test coordinates (normalized)
        image_size: Size of images
    """
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Convert to pixel coordinates
    y_test_pixels = y_test * (image_size - 1)
    y_pred_pixels = y_pred * (image_size - 1)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_pixels, y_pred_pixels)
    mae = mean_absolute_error(y_test_pixels, y_pred_pixels)
    
    # Euclidean distance - direct point-to-point distance
    euclidean_distances = np.sqrt(
        np.sum((y_test_pixels - y_pred_pixels)**2, axis=1)
    )
    avg_distance = np.mean(euclidean_distances)
    median_distance = np.median(euclidean_distances)
    std_distance = np.std(euclidean_distances)
    
    # Accuracy thresholds
    within_1px = np.mean(euclidean_distances <= 1.0) * 100
    within_2px = np.mean(euclidean_distances <= 2.0) * 100
    within_3px = np.mean(euclidean_distances <= 3.0) * 100
    within_5px = np.mean(euclidean_distances <= 5.0) * 100
    
    # Print results
    print("\n" + "="*70)
    print("TEST SET EVALUATION RESULTS")
    print("="*70)
    print(f"\nPrimary Metrics:")
    print(f"  MSE (Mean Squared Error):      {mse:.4f} pixels²")
    print(f"  MAE (Mean Absolute Error):     {mae:.4f} pixels")
    
    print(f"\nEuclidean Distance (point-to-point distance):")
    print(f"  Mean:                          {avg_distance:.4f} pixels")
    print(f"  Median:                        {median_distance:.4f} pixels")
    print(f"  Std Dev:                       {std_distance:.4f} pixels")
    print(f"  Range:                         [{np.min(euclidean_distances):.4f}, {np.max(euclidean_distances):.4f}] pixels")
    
    print(f"\nAccuracy Thresholds:")
    print(f"  Within 1 pixel:                {within_1px:.2f}%")
    print(f"  Within 2 pixels:               {within_2px:.2f}%")
    print(f"  Within 3 pixels:               {within_3px:.2f}%")
    print(f"  Within 5 pixels:               {within_5px:.2f}%")
    print("="*70 + "\n")


def plot_training_history(history: keras.callbacks.History, 
                         save_path: str = 'training_curves.png') -> None:
    """
    Plot and save training history.
    
    Args:
        history: Keras training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss (MSE)
    axes[0].plot(history.history['loss'], label='Training Loss (MSE)', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss (MSE)', 
                linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', 
                linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error (normalized)', fontsize=12)
    axes[1].set_title('Model MAE Over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def visualize_predictions(model: keras.Model,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         num_samples: int = 10,
                         image_size: int = 50,
                         save_path: str = 'predictions.png') -> None:
    """
    Visualize and save model predictions.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test coordinates (normalized)
        num_samples: Number of samples to visualize
        image_size: Size of images
        save_path: Path to save the plot
    """
    # Make predictions
    y_pred = model.predict(X_test[:num_samples], verbose=0)
    
    # Convert to pixel coordinates
    true_pixels = y_test[:num_samples] * (image_size - 1)
    pred_pixels = y_pred * (image_size - 1)
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()
    
    for i in range(num_samples):
        true_x, true_y = true_pixels[i]
        pred_x, pred_y = pred_pixels[i]
        
        # Calculate Euclidean distance error
        error = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
        
        # Display image
        axes[i].imshow(X_test[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        
        # Mark true position (green circle)
        axes[i].plot(true_x, true_y, 'go', markersize=10, 
                    fillstyle='none', markeredgewidth=2, label='True')
        
        # Mark predicted position (red X)
        axes[i].plot(pred_x, pred_y, 'rx', markersize=10, 
                    markeredgewidth=2, label='Predicted')
        
        # Draw line
        axes[i].plot([true_x, pred_x], [true_y, pred_y], 
                    'b--', linewidth=1, alpha=0.5)
        
        axes[i].set_title(f'True: ({int(true_x)}, {int(true_y)})\\n'
                         f'Pred: ({int(pred_x)}, {int(pred_y)})\\n'
                         f'Error: {error:.2f}px', fontsize=9)
        axes[i].axis('off')
        
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Predictions vs Ground Truth (Green=True, Red=Predicted)', 
                y=1.02, fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions visualization saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    
    # Configuration
    NUM_SAMPLES = 7500  # 3× coverage of 2,500 positions
    IMAGE_SIZE = 50
    BATCH_SIZE = 32
    EPOCHS = 20
    
    print("="*70)
    print("PIXEL COORDINATE PREDICTION - OPTIMIZED LIGHTWEIGHT CNN")
    print("="*70)
    
    # Generate dataset
    print("\n1. Generating dataset...")
    X, y = generate_dataset(NUM_SAMPLES, IMAGE_SIZE)
    print(f"   Generated {NUM_SAMPLES} samples")
    print(f"   Rationale: {NUM_SAMPLES}/{IMAGE_SIZE*IMAGE_SIZE} = "
          f"{NUM_SAMPLES/(IMAGE_SIZE*IMAGE_SIZE):.1f}× coverage")
    print(f"   Image shape: {X.shape}")
    print(f"   Coordinates shape: {y.shape}")
    
    # Split data (70/15/15)
    print("\n2. Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"   Training: {len(X_train)} samples (70%)")
    print(f"   Validation: {len(X_val)} samples (15%)")
    print(f"   Test: {len(X_test)} samples (15%)")
    
    # Build model
    print("\n3. Building lightweight CNN model...")
    model = build_lightweight_cnn((IMAGE_SIZE, IMAGE_SIZE, 1))
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Architecture: Conv(8)→Pool→Conv(16)→Pool→Dense(32)→Output(2)")
    
    # Setup callbacks
    callback_list = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, 
            min_lr=1e-7, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=5, 
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_model_optimized.keras', monitor='val_loss',
            save_best_only=True, verbose=0
        )
    ]
    
    # Train model
    print("\n4. Training model...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max epochs: {EPOCHS}")
    print(f"   Loss: MSE, Metric: MAE")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callback_list,
        verbose=1
    )
    
    # Plot training curves
    print("\n5. Generating visualizations...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n6. Evaluating on test set...")
    evaluate_model(model, X_test, y_test, IMAGE_SIZE)
    
    # Visualize predictions
    visualize_predictions(model, X_test, y_test, num_samples=10, 
                         image_size=IMAGE_SIZE)
    
    # Save final model
    print("\n7. Saving final model...")
    model.save('pixel_coordinate_model_optimized_final.keras')
    print("   Model saved as 'pixel_coordinate_model_optimized_final.keras'")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - best_model_optimized.keras")
    print("  - pixel_coordinate_model_optimized_final.keras")
    print("  - training_curves.png")
    print("  - predictions.png")
    print("\nDataset Design Summary:")
    print(f"  • {NUM_SAMPLES} samples (3× coverage of all positions)")
    print(f"  • Uniform random distribution")
    print(f"  • Normalization: images [0,1], coords [0,1] via x/49, y/49")
    print("\nModel Architecture:")
    print(f"  • Lightweight CNN: ~{model.count_params():,} parameters")
    print(f"  • Two conv blocks (8, 16 filters)")
    print(f"  • Optimized for single-pixel detection")
    print("\nMetrics Used:")
    print(f"  • MSE (Mean Squared Error) - primary loss metric")
    print(f"  • MAE (Mean Absolute Error) - average pixel error")
    print(f"  • Euclidean Distance - direct point-to-point distance")


if __name__ == "__main__":
    main()
