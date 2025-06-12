# 🌤️ Weather Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/code/reimei2009/weatherclassfication)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive deep learning project for classifying weather conditions from images using a custom **AdvancedWeatherCNN** architecture with **PyTorch**. The project explores various train-test split ratios, performs PCA/LDA visualizations, evaluates classical machine learning models, and applies K-Means clustering for feature analysis.

## 🎯 Project Highlights

- **Custom CNN Architecture**: Designed robust AdvancedWeatherCNN for weather image classification
- **Multiple Train-Test Ratios**: Evaluates performance across 80:20, 70:30, and 60:40 splits
- **Feature Analysis**: PCA and LDA for dimensionality reduction and visualization
- **ML Integration**: Comparative evaluation with Logistic Regression, SVM, KNN, and ANN
- **Clustering Analysis**: K-Means clustering to explore feature distributions
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and confusion matrices

## 📊 Demo & Results

🔗 **Live Demo**: [Weather Classification Notebook on Kaggle](https://www.kaggle.com/code/reimei2009/weatherclassfication)

## 🗂️ Dataset

The dataset is sourced from Kaggle: [Weather Dataset](https://www.kaggle.com/datasets/nikhilshingadiya/weather-image-recognition)

**Dataset Structure:**
- Images organized by weather categories (sunny, rainy, cloudy, foggy, snowy)
- Balanced distribution across classes
- High-quality weather images suitable for classification tasks
- Comprehensive coverage of various weather conditions

## 🏗️ Project Architecture

```
weather_classifier/
├── 📁 src/                         # Source code
│   ├── 📄 config.py                # Configuration & hyperparameters
│   ├── 📁 data/                    # Data handling
│   │   └── data_loader.py          # Dataset loading & DataLoaders
│   ├── 📁 models/                  # Model architecture
│   │   ├── model.py                # AdvancedWeatherCNN definition
│   │   └── train.py                # Training & evaluation logic
│   ├── 📁 utils/                   # Utility functions
│   │   ├── visualization.py        # EDA, PCA/LDA, confusion matrices
│   │   ├── ml_evaluation.py        # ML model evaluations
│   │   └── clustering.py           # K-Means clustering
│   └── 📄 main.py                  # Main pipeline orchestration
├── 📁 models/                      # Saved model checkpoints
├── 📁 sample/                      # Visualizations & plots
├── 📄 requirements.txt             # Dependencies
└── 📄 README.md                   # Documentation
```

### 📋 Component Details

| Component | Description |
|-----------|-------------|
| `config.py` | Global constants, hyperparameters, and data transformation settings |
| `data/data_loader.py` | Dataset loading, train/test splitting, and DataLoader creation |
| `models/model.py` | Custom AdvancedWeatherCNN architecture for weather classification |
| `models/train.py` | Training loop, evaluation, early stopping, and model checkpointing |
| `utils/visualization.py` | Image display, PCA/LDA scatter plots, confusion matrices, and performance comparisons |
| `utils/ml_evaluation.py` | Evaluation of Logistic Regression, SVM, KNN, and ANN on PCA/LDA features |
| `utils/clustering.py` | K-Means clustering on PCA/LDA features with silhouette score analysis |
| `main.py` | Orchestrates the complete pipeline from training to evaluation |

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended) or CPU
- **8GB+ RAM**
- **Sufficient storage** for dataset and models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/reimei2009/weather_classifier.git
   cd weather_classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup dataset**
   - Download from [Kaggle Weather Dataset](https://www.kaggle.com/datasets/nikhilshingadiya/weather-image-recognition)
   - Place in `./data/weather_dataset/dataset`
   - Ensure the folder structure matches the dataset requirements

## 🎮 Usage

### Training the Model

Start training with default parameters:

```bash
python src/main.py
```

**Customization Options:**
- Modify train-test ratios and hyperparameters in `src/config.py`
- Adjust data augmentation in `config.py`
- Monitor training progress through saved plots and logs

### Making Predictions

```python
from src.models.model import AdvancedWeatherCNN
from src.utils.visualization import predict_image
import torch

# Load model
model = AdvancedWeatherCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("models/best_acc_model_80_20.pth"))
model.eval()

# Predict single image
result = predict_image(
    image_path="path/to/your/image.jpg",
    model=model,
    device=device
)
print(f"Prediction: {result}")
```

### Hyperparameter Tuning

Key parameters to experiment with:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.0001 | Controls training speed |
| `BATCH_SIZE` | 8 | Samples per batch |
| `NUM_EPOCHS` | 20 | Training iterations |
| `IMG_SIZE` | 224 | Input image dimensions |

## 🧠 Model Architecture

### AdvancedWeatherCNN Overview

AdvancedWeatherCNN is a custom convolutional neural network featuring:

- **Convolutional Layers**: Four Conv2D layers (32, 64, 128, 256 filters) with batch normalization
- **Pooling**: Max pooling for dimensionality reduction
- **Fully Connected Layers**: Two dense layers with dropout for regularization
- **Activation**: ReLU for non-linearity
- **Output**: Softmax for multi-class classification

### Training Strategy

1. **Data Preprocessing**: Normalization and augmentation (random flips, rotations, color jitter)
2. **Optimization**: Adam optimizer with learning rate scheduling
3. **Regularization**: Early stopping, dropout, and weight decay
4. **Evaluation**: Multi-metric assessment (accuracy, precision, recall, loss)
5. **Feature Extraction**: PCA and LDA for visualization and ML model input

## 📈 Performance Monitoring

The project tracks multiple metrics:

- **Training/Validation Loss**: Monitors model convergence
- **Accuracy**: Classification performance
- **Precision/Recall**: Balanced evaluation for multi-class tasks
- **Confusion Matrix**: Per-class performance analysis
- **Silhouette Score**: Clustering quality for K-Means

All metrics and visualizations are saved in the `sample/` directory.

## ☁️ Training on Kaggle

For users without local GPU resources:

1. **Access the [Kaggle Notebook](https://www.kaggle.com/code/reimei2009/weatherclassfication)**
2. **Fork the notebook** to your account
3. **Enable GPU acceleration** in notebook settings
4. **Run all cells** to start training
5. **Download trained models** for local inference

## 🔧 Advanced Configuration

### Custom Data Augmentation

```python
# In config.py
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    # Add your custom transforms here
])
```

### Model Checkpointing

The system automatically saves:
- **Best Loss Model**: Lowest validation loss
- **Best Accuracy Model**: Highest validation accuracy
- **Epoch Checkpoints**: Models saved per epoch for each ratio

### Multi-Ratio Evaluation

The project evaluates three different train-test splits:
- **80:20 Split**: Maximum training data for best performance
- **70:30 Split**: Balanced approach for robust evaluation
- **60:40 Split**: Limited training data to test generalization

## 🔬 Feature Analysis

### Dimensionality Reduction

- **PCA**: Principal Component Analysis for feature visualization
- **LDA**: Linear Discriminant Analysis for class separation
- **Scatter Plots**: 2D and 3D visualizations of reduced features

### Machine Learning Comparison

Comparative evaluation of classical ML models:
- **Logistic Regression**: Linear classification baseline
- **Support Vector Machine**: Non-linear classification with RBF kernel
- **K-Nearest Neighbors**: Instance-based learning
- **Artificial Neural Network**: Multi-layer perceptron

### Clustering Analysis

- **K-Means Clustering**: Unsupervised grouping of features
- **Silhouette Analysis**: Quality assessment of clustering
- **Visualization**: Cluster boundaries and centroids

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or suggest features
- Submit pull requests
- Improve documentation
- Add new weather categories or datasets

## 👨‍💻 Author

**Ngô Thanh Tình (reimei2009)**
- 🐙 GitHub: [@reimei2009](https://github.com/reimei2009)
- 📧 Email: thanhin875@gmail.com
- 🏆 Kaggle: [reimei2009](https://www.kaggle.com/reimei2009)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Provided by the Kaggle community
- **Framework**: Built with PyTorch, torchvision, and scikit-learn
- **Inspiration**: Modern computer vision and machine learning techniques
- **Community**: Kaggle community for feedback and support

## 📚 References

- [CNN Architectures](https://arxiv.org/abs/1409.1556) - Very Deep Convolutional Networks for Large-Scale Image Recognition
- [PyTorch Documentation](https://pytorch.org/docs/) - Official PyTorch documentation
- [PCA and LDA](https://scikit-learn.org/stable/modules/decomposition.html) - Scikit-learn dimensionality reduction
- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means) - Scikit-learn clustering documentation

---

<div align="center">

**🌞 Happy Weather Classification! 🌧️**

*If you found this project helpful, please consider giving it a ⭐*

</div>
