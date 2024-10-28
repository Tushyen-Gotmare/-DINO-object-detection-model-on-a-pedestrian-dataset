import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
from matplotlib import patches

# Cache the function to improve performance
@st.cache_data
def draw_bounding_boxes(image_input, boxes, scores=None, threshold=0.5):
    """Draw bounding boxes on the image with optional confidence scores"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image_input, Image.Image):
        image = np.array(image_input)
    else:
        image = image_input
        
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    height, width = image.shape[:2]
    
    for idx, box in enumerate(boxes):
        if scores is None or scores[idx] >= threshold:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            if scores is not None:
                conf_text = f'{scores[idx]:.2f}'
                ax.text(x1, y1-5, conf_text, color='red', fontsize=8)
    
    plt.axis('off')
    return fig

@st.cache_data
def create_sample_metrics():
    """Create sample metrics for demonstration"""
    return {
        'AP@0.5': 0.85,
        'AP@0.75': 0.70,
        'AP@0.5:0.95': 0.65,
        'AP (small)': 0.55,
        'AP (medium)': 0.75,
        'AP (large)': 0.80
    }

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
        # Create placeholder for progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        plot_placeholder = st.empty()
        
        # Initialize metrics
        train_losses = []
        val_losses = []
        
        # Simulated training loop with better error handling
        try:
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
                fig.update_layout(
                    title='Training Progress',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    height=400
                )
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                
            st.success("Training Complete!")
        except Exception as e:
            st.error(f"An error occurred during training: {str(e)}")

def evaluation_analysis_page():
    st.header("Evaluation & Analysis")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Precision Metrics")
            ap_data = create_sample_metrics()
            
            fig = go.Figure(data=[
                go.Bar(x=list(ap_data.keys()), 
                      y=list(ap_data.values()),
                      text=[f'{v:.2%}' for v in ap_data.values()],
                      textposition='auto')
            ])
            fig.update_layout(
                title='Average Precision Metrics',
                yaxis_range=[0,1],
                yaxis_title='AP Score',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
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
                yaxis_title='Count',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = np.array([[85, 15], [12, 88]])
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500,
            width=500
        )
        
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred in evaluation analysis: {str(e)}")

@st.cache_data
def load_and_process_image(uploaded_file):
    """Load and process uploaded image with caching"""
    return Image.open(uploaded_file)

def attention_maps_page():
    st.header("Attention Maps Visualization")
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            image = load_and_process_image(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image with Detections")
                st.image(image, caption='Input Image', use_column_width=True)
                
                # Sample boxes for demonstration
                sample_boxes = [[100, 100, 200, 300], [300, 150, 400, 350]]
                sample_scores = [0.95, 0.87]
                
                fig = draw_bounding_boxes(image, sample_boxes, sample_scores)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Attention Maps")
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
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def dataset_visualization_page():
    st.header("Dataset Visualization")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            # Sample dataset statistics
            dataset_stats = {
                'Training': 160,
                'Validation': 40
            }
            
            fig = go.Figure(data=[
                go.Pie(labels=list(dataset_stats.keys()),
                      values=list(dataset_stats.values()),
                      hole=.3)
            ])
            fig.update_layout(title="Dataset Split", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            total_images = sum(dataset_stats.values())
            st.write("Dataset Overview:")
            st.write(f"- Total Images: {total_images}")
            for key, value in dataset_stats.items():
                st.write(f"- {key} Images: {value}")
        
        with col2:
            st.subheader("Image Visualization")
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = load_and_process_image(uploaded_file)
                st.image(image, caption='Original Image', use_column_width=True)
                
                # Display image information
                width, height = image.size
                st.write(f"- Dimensions: {width} x {height}")
                
    except Exception as e:
        st.error(f"An error occurred in dataset visualization: {str(e)}")

def main():
    st.set_page_config(
        page_title="DINO Pedestrian Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("DINO Pedestrian Detection Analysis Dashboard")
    
    # Add sidebar configuration
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select a Page",
        ["Dataset Visualization", "Model Training", "Evaluation & Analysis", "Attention Maps"]
    )
    
    # Add configuration options
    st.sidebar.subheader("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Page routing
    try:
        if page == "Dataset Visualization":
            dataset_visualization_page()
        elif page == "Model Training":
            model_training_page()
        elif page == "Evaluation & Analysis":
            evaluation_analysis_page()
        elif page == "Attention Maps":
            attention_maps_page()
    except Exception as e:
        st.error(f"An error occurred while loading the page: {str(e)}")

if __name__ == "__main__":
    main()
