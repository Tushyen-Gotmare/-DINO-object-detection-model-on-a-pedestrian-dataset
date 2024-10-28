import streamlit as st
import torch
import numpy as np
import pandas as pd
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import io
import matplotlib.patches as patches

def draw_bounding_boxes(image, boxes, scores=None, threshold=0.5):
    """Draw bounding boxes on the image with optional confidence scores"""
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    height, width = image_np.shape[:2]
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        if scores is not None and scores[idx] >= threshold:
            conf_text = f'{scores[idx]:.2f}'
            ax.text(x1, y1-5, conf_text, color='red', fontsize=8)
    
    plt.axis('off')
    return fig

def model_training_page():
    st.header("Model Training")
    
    # Training parameters
    st.subheader("Training Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=8)
        epochs = st.slider("Number of Epochs", min_value=1, max_value=100, value=10)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, 
                                      value=0.001, format="%.4f")
    
    with col2:
        optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "AdamW"])
        scheduler = st.selectbox("Learning Rate Scheduler", 
                               ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, 
                                     value=0.0001, format="%.4f")
    
    # Training progress
    if st.button("Start Training"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training metrics plot
        plot_placeholder = st.empty()
        fig = go.Figure()
        
        # Simulated training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Simulate training and validation loss
            train_loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
            val_loss = 1.2 / (epoch + 1) + np.random.random() * 0.1
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs}")
            
            # Update plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=train_losses, name='Train Loss'))
            fig.add_trace(go.Scatter(y=val_losses, name='Validation Loss'))
            fig.update_layout(title='Training Progress',
                            xaxis_title='Epoch',
                            yaxis_title='Loss')
            plot_placeholder.plotly_chart(fig)
        
        st.success("Training Complete!")

def evaluation_analysis_page():
    st.header("Evaluation & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Precision Metrics")
        # AP metrics visualization
        ap_data = {
            'AP@0.5': 0.85,
            'AP@0.75': 0.70,
            'AP@0.5:0.95': 0.65,
            'AP (small)': 0.55,
            'AP (medium)': 0.75,
            'AP (large)': 0.80
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(ap_data.keys()), 
                  y=list(ap_data.values()),
                  text=[f'{v:.2%}' for v in ap_data.values()],
                  textposition='auto')
        ])
        fig.update_layout(
            title='Average Precision Metrics',
            yaxis_range=[0,1],
            yaxis_title='AP Score'
        )
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Error Analysis")
        error_types = {
            'False Positives': 15,
            'False Negatives': 12,
            'Poor Localization': 8,
            'Scale Errors': 5,
            'Occlusion Cases': 7
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(error_types.keys()), 
                  y=list(error_types.values()),
                  text=list(error_types.values()),
                  textposition='auto')
        ])
        fig.update_layout(
            title='Error Distribution',
            yaxis_title='Count'
        )
        st.plotly_chart(fig)
    
    # Confusion Matrix with fixed visualization
    st.subheader("Confusion Matrix")
    conf_matrix = np.array([[85, 15], [12, 88]])
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Negative', 'Positive'],
        y=['Negative', 'Positive'],
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500
    )
    
    st.plotly_chart(fig)
    
    # Performance across different scenarios
    st.subheader("Performance Across Scenarios")
    scenarios = {
        'Normal': 0.88,
        'Crowded': 0.75,
        'Low Light': 0.70,
        'Partial Occlusion': 0.65,
        'Far Distance': 0.60
    }
    
    fig = go.Figure(data=[
        go.Bar(x=list(scenarios.keys()),
               y=list(scenarios.values()),
               text=[f'{v:.2%}' for v in scenarios.values()],
               textposition='auto')
    ])
    fig.update_layout(
        title='Performance by Scenario',
        yaxis_range=[0,1],
        yaxis_title='Detection Rate'
    )
    st.plotly_chart(fig)

    # Add detailed metrics breakdown
    st.subheader("Detailed Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Size-based Performance:")
        size_metrics = pd.DataFrame({
            'Size': ['Small', 'Medium', 'Large'],
            'AP': [0.55, 0.75, 0.80],
            'Recall': [0.50, 0.70, 0.85]
        })
        st.dataframe(size_metrics)
    
    with col2:
        st.write("Occlusion Performance:")
        occlusion_metrics = pd.DataFrame({
            'Occlusion': ['None', 'Partial', 'Heavy'],
            'AP': [0.85, 0.65, 0.45],
            'Recall': [0.90, 0.60, 0.40]
        })
        st.dataframe(occlusion_metrics)

def attention_maps_page():
    st.header("Attention Maps Visualization")
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image with Detections")
            st.image(image, caption='Input Image')
            
            # Sample boxes for demonstration
            sample_boxes = [[100, 100, 200, 300], [300, 150, 400, 350]]
            sample_scores = [0.95, 0.87]
            
            fig = draw_bounding_boxes(image, sample_boxes, sample_scores)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Attention Maps")
            
            # Layer selection
            layer_num = st.slider("Select Attention Layer", 1, 6, 1)
            
            # Generate sample attention map
            attention_map = np.random.rand(100, 100)
            fig, ax = plt.subplots()
            sns.heatmap(attention_map, ax=ax, cmap='viridis')
            plt.title(f'Attention Map - Layer {layer_num}')
            st.pyplot(fig)
            
            # Add attention statistics
            st.write("Attention Statistics:")
            st.write(f"- Max Attention Value: {attention_map.max():.3f}")
            st.write(f"- Mean Attention Value: {attention_map.mean():.3f}")
            st.write(f"- Attention Entropy: {-(attention_map * np.log(attention_map + 1e-10)).sum():.3f}")

def dataset_visualization_page():
    st.header("Dataset Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Statistics")
        total_images = 200
        train_images = 160
        val_images = 40
        
        fig = go.Figure(data=[
            go.Pie(labels=['Training', 'Validation'],
                  values=[train_images, val_images],
                  hole=.3)
        ])
        fig.update_layout(title="Dataset Split")
        st.plotly_chart(fig)
        
        st.write("Dataset Overview:")
        st.write(f"- Total Images: {total_images}")
        st.write(f"- Training Images: {train_images}")
        st.write(f"- Validation Images: {val_images}")
    
    with col2:
        st.subheader("Image Visualization")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Original Image', use_column_width=True)
            
            sample_boxes = [
                [100, 100, 200, 300],
                [300, 150, 400, 350]
            ]
            sample_scores = [0.95, 0.87]
            
            st.subheader("Detection Visualization")
            fig = draw_bounding_boxes(image, sample_boxes, sample_scores)
            st.pyplot(fig)
            
            st.write("Image Information:")
            width, height = image.size
            st.write(f"- Dimensions: {width} x {height}")
            st.write(f"- Number of detected pedestrians: {len(sample_boxes)}")
            
            if len(sample_scores) > 0:
                st.subheader("Confidence Score Distribution")
                fig = go.Figure(data=[
                    go.Histogram(x=sample_scores, nbinsx=10)
                ])
                fig.update_layout(
                    title="Detection Confidence Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="DINO Pedestrian Detection", layout="wide")
    
    st.title("DINO Pedestrian Detection Analysis Dashboard")
    
    page = st.sidebar.selectbox(
        "Select a Page",
        ["Dataset Visualization", "Model Training", "Evaluation & Analysis", "Attention Maps"]
    )
    
    st.sidebar.subheader("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    if page == "Dataset Visualization":
        dataset_visualization_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Evaluation & Analysis":
        evaluation_analysis_page()
    elif page == "Attention Maps":
        attention_maps_page()

if __name__ == "__main__":
    main()