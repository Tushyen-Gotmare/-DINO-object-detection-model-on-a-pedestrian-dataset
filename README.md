# DINO Pedestrian Detection Dashboard

## Overview
This project implements an interactive dashboard for pedestrian detection using the DINO (DETR with Improved deNoising anchOr boxes) model. The application provides comprehensive visualization and analysis tools for training, evaluating, and understanding pedestrian detection results on the IIT Delhi campus dataset.

## Features
- **Dataset Visualization**
  - Interactive visualization of dataset statistics
  - Image annotation viewer
  - Data distribution analysis
  - Training/validation split overview

- **Model Training**
  - Real-time training progress monitoring
  - Configurable hyperparameters
  - Loss curve visualization
  - Training metrics tracking
  - Multiple optimizer options
  - Learning rate scheduler selection

- **Evaluation & Analysis**
  - Average Precision (AP) metrics visualization
  - Error analysis breakdown
  - Confusion matrix
  - Performance across different scenarios
  - Detailed metrics by object size and occlusion

- **Attention Maps**
  - Layer-wise attention visualization
  - Interactive attention map analysis
  - Attention statistics
  - Side-by-side comparison with original images

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/dino-pedestrian-detection.git
cd dino-pedestrian-detection
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Packages
```plaintext
streamlit
torch
torchvision
numpy
opencv-python
pillow
matplotlib
seaborn
plotly
pandas
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Navigate to the different pages using the sidebar:
   - Dataset Visualization
   - Model Training
   - Evaluation & Analysis
   - Attention Maps

3. Upload images and configure parameters as needed.

## Dataset
The application is designed to work with the IIT Delhi campus pedestrian dataset:
- 200 total images
- 160 training images
- 40 validation images
- COCO format annotations

## Model Details
- Base Model: DINO (DETR with Improved deNoising anchOr boxes)
- Backbone: ResNet-50
- Pre-trained weights: COCO dataset
- Fine-tuned on pedestrian detection

## Project Structure
```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── data/                  # Dataset directory
│   ├── train/            # Training images
│   ├── val/              # Validation images
│   └── annotations/      # COCO format annotations
└── models/               # Model weights and configurations
    ├── pretrained/      # Pre-trained DINO weights
    └── finetuned/       # Fine-tuned model weights
```

## Key Functions

### Dataset Visualization
- `dataset_visualization_page()`: Displays dataset statistics and sample images
- `draw_bounding_boxes()`: Visualizes detection boxes on images

### Model Training
- `model_training_page()`: Handles model training interface and progress tracking
- Training configuration options:
  - Batch size
  - Learning rate
  - Optimizer selection
  - Scheduler options

### Evaluation & Analysis
- `evaluation_analysis_page()`: Provides comprehensive evaluation metrics
- Visualizations include:
  - AP metrics
  - Error analysis
  - Confusion matrix
  - Scenario-based performance

### Attention Maps
- `attention_maps_page()`: Visualizes model attention patterns
- Features:
  - Layer-wise attention visualization
  - Attention statistics
  - Interactive layer selection

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Improvements
- [ ] Add real-time video detection
- [ ] Implement multi-GPU training support
- [ ] Add model export functionality
- [ ] Enhance attention visualization
- [ ] Add cross-validation support
- [ ] Implement automated hyperparameter tuning


## Acknowledgments
- DINO model implementation
- IIT Delhi campus dataset
- Streamlit framework
- PyTorch ecosystem

## Contact
Mail ID- tushyengotmare.email@gmail.com
Project Link: 
