# Pixel Coordinate Prediction using Deep Learning

## Problem Statement
Using Deep Learning techniques, predict the coordinates (x, y) of a single white pixel (value 255) in a 50x50 grayscale image where all other pixels are 0.

## Solution Overview
This project implements an **optimized lightweight Convolutional Neural Network (CNN)** to solve the pixel coordinate regression problem. The architecture is specifically designed for efficient single-pixel detection with minimal parameters while maintaining high accuracy.

## Key Features
- ✅ Custom dataset generation with uniform random distribution
- ✅ CNN architecture optimized for spatial regression
- ✅ Comprehensive training pipeline with callbacks
- ✅ Detailed visualizations and error analysis
- ✅ PEP8 compliant code with extensive comments
- ✅ Multiple evaluation metrics

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Extract the Repository
```bash
# If using git
git clone <repository-url>
cd pixel-coordinate-prediction

# If using compressed file
unzip pixel-coordinate-prediction.zip
cd pixel-coordinate-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

Then open `pixel_coordinate_prediction.ipynb` in your browser.

## Project Structure
```
.
├── pixel_coordinate_prediction.ipynb  # Main notebook with complete solution
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── best_model.keras                  # Saved best model (after training)
├── pixel_coordinate_model_final.keras # Final trained model (after training)
├── training_history.pkl              # Training metrics (after training)
└── test_predictions.npz              # Test predictions (after training)
```

## Dataset Rationale

### Why 7,500 Samples (3× Coverage)?
1. **Total possible positions**: 50 × 50 = 2,500 unique coordinates
2. **7,500 samples ≈ 3× coverage** of all positions
3. **Ensures uniform learning** without excessive redundancy
4. **Computational efficiency**: Fast training while providing sufficient data
5. **Professional rationale**: "Since the image contains only one active pixel and 2,500 unique positions exist, 7,500 samples (≈3× coverage) were generated to ensure uniform learning without excessive redundancy."

### Dataset Specifications
- **Total Samples**: 7,500 (3× coverage of 2,500 possible positions)
- **Image Size**: 50x50 pixels
- **Distribution**: Uniform random across all positions
- **Train/Val/Test Split**: 70% / 15% / 15% (5,250 / 1,125 / 1,125)
- **Normalization**: 
  - Images: [0, 255] → [0, 1] (divide by 255)
  - Coordinates: [0, 49] → [0, 1] (divide by 49: x/49, y/49)

## Model Architecture

### Lightweight CNN Design
```
Input (50, 50, 1)
    ↓
Conv2D(8, 3x3, padding='same') + ReLU
    ↓
MaxPooling(2x2) → (25, 25, 8)
    ↓
Conv2D(16, 3x3) + ReLU
    ↓
MaxPooling(2x2) → (12, 12, 16)
    ↓
Flatten → (2304)
    ↓
Dense(32) + ReLU
    ↓
Dense(2) → (x, y) coordinates
```

**Total Parameters**: ~75,000 (very lightweight!)

### Why This Architecture?
1. **Optimized for Simple Task**: Single-pixel detection doesn't require deep networks
2. **Prevents Overfitting**: Lightweight design with minimal parameters
3. **Fast Training**: Converges in ~10-15 epochs
4. **Efficient Feature Extraction**: Two conv blocks capture spatial patterns at different scales
5. **No Heavy Regularization**: Clean dataset doesn't require dropout or BatchNorm
6. **Industry Appropriate**: Demonstrates CNN understanding while avoiding unnecessary complexity

## Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: MSE (Mean Squared Error)
- **Metrics**: MAE (Mean Absolute Error)
- **Batch Size**: 32
- **Max Epochs**: 20 (typically converges in 10-15)
- **Callbacks**:
  - ReduceLROnPlateau (factor=0.5, patience=3)
  - EarlyStopping (patience=5)
  - ModelCheckpoint (save best model)

## Evaluation Metrics

### Primary Metrics
1. **MSE (Mean Squared Error)**: 
   - Matches the loss function
   - Standard regression metric
   - Penalizes larger errors more heavily

2. **MAE (Mean Absolute Error)**: 
   - Average pixel error
   - Easy to interpret (e.g., MAE = 0.4 means < 1 pixel error average)
   - Intuitive for reports

3. **Euclidean Distance** (bonus metric):
   - Formula: `sqrt((x_pred - x_true)² + (y_pred - y_true)²)`
   - Direct geometric distance between predicted and actual points
   - Most meaningful for coordinate prediction
   - Reports mean, median, and standard deviation

### Accuracy Thresholds
- % of predictions within 1 pixel
- % of predictions within 2 pixels  
- % of predictions within 3 pixels
- % of predictions within 5 pixels

## Results Visualization
The notebook includes:
- Sample dataset images with annotations
- Coordinate distribution analysis
- Training/validation loss curves
- Prediction vs ground truth comparisons
- Comprehensive error analysis with multiple plots
- Error distribution histograms

## Usage

### Running the Complete Pipeline
Simply execute all cells in the notebook sequentially. The notebook will:
1. Generate the dataset
2. Split into train/val/test sets
3. Build and compile the CNN model
4. Train with callbacks and monitoring
5. Evaluate on test set
6. Generate comprehensive visualizations
7. Save models and results

### Using the Trained Model
```python
import numpy as np
from tensorflow import keras

# Load the model
model = keras.models.load_model('pixel_coordinate_model_final.keras')

# Make predictions on new images
# images should be shape (n, 50, 50, 1) with values in [0, 1]
predictions = model.predict(images)

# Convert normalized predictions to pixel coordinates
pixel_coords = predictions * 49  # Scale back to [0, 49]
```


## Why This Approach is Better

### Advantages of CNN over Alternatives:
1. **Fully Connected Networks**: CNNs use fewer parameters and learn spatial features
2. **Classification Approach**: Regression is more natural for continuous coordinates
3. **Manual Feature Engineering**: CNNs automatically learn optimal features

### Technical Decisions:
- **Normalized coordinates**: Improves gradient descent stability
- **Sigmoid activation**: Constrains output to [0, 1] range
- **BatchNormalization**: Accelerates training and improves convergence
- **Dropout**: Prevents overfitting on relatively simple task

## Dependencies Explanation
- **tensorflow**: Deep learning framework for building and training the CNN
- **numpy**: Numerical computations and array operations
- **matplotlib**: Visualization of results and training curves
- **seaborn**: Enhanced statistical visualizations
- **scikit-learn**: Train/test splitting and evaluation metrics
- **jupyter**: Interactive notebook environment

## Expected Performance
With the optimized lightweight configuration, you should expect:
- **Training Time**: 2-5 minutes (depending on hardware)
- **Convergence**: Within 10-15 epochs
- **Average Error**: < 1 pixel Euclidean distance
- **Within 1 Pixel**: > 60% accuracy
- **Within 3 Pixels**: > 95% accuracy
- **Model Size**: ~300 KB (very lightweight!)

## Troubleshooting

### Common Issues:

**1. TensorFlow Installation Issues**
```bash
# Try installing with specific version
pip install tensorflow==2.15.0

# For Apple Silicon Macs
pip install tensorflow-macos==2.15.0
```

**2. Memory Issues**
- Reduce `BATCH_SIZE` from 32 to 16
- Reduce `NUM_SAMPLES` from 15000 to 10000

**3. Jupyter Kernel Issues**
```bash
python -m ipykernel install --user --name=venv
```

## Future Improvements
1. Experiment with attention mechanisms
2. Try different architectures (ResNet, EfficientNet)
3. Implement data augmentation (rotation, flipping)
4. Add ensemble methods
5. Optimize for edge/corner pixel positions

## Author
Aryan Kumar

## License
This project is for educational purposes.

## Acknowledgments
- TensorFlow/Keras documentation
- Deep learning best practices from academic literature
