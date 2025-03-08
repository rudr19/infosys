import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, Xception
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import io
import zipfile
import shutil
import time
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Advanced Image Classification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

# Configuration for dataset-based training
class Config:
    # Image parameters
    IMG_SIZE = 224
    CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    
    # Model parameters
    DROPOUT_RATE = 0.3
    
    # Paths
    MODEL_PATH = 'image_classification_model.h5'
    DATASET_PATH = 'dataset'  # Will be created if needed
    NUM_CLASSES = None  # Will be set after dataset processing
    
config = Config()

# Dataset handling
class DatasetHandler:
    def __init__(self, config):
        self.config = config
        
    def process_uploaded_zip(self, uploaded_zip):
        """Process uploaded zip file containing dataset"""
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save the uploaded zip file
            zip_path = os.path.join(tmp_dir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            
            # Ensure dataset directory exists
            if os.path.exists(self.config.DATASET_PATH):
                shutil.rmtree(self.config.DATASET_PATH)
            os.makedirs(self.config.DATASET_PATH)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.DATASET_PATH)
            
            st.success(f"Dataset extracted to {self.config.DATASET_PATH}")
            return True
    
    def process_dataset(self):
        """Process the dataset and create data generators"""
        # Create data generators with augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=efficientnet_preprocess,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Create generators for training and validation
        train_generator = train_datagen.flow_from_directory(
            self.config.DATASET_PATH,
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            self.config.DATASET_PATH,
            target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Get class names and update NUM_CLASSES
        class_names = list(train_generator.class_indices.keys())
        self.config.NUM_CLASSES = len(class_names)
        st.write(f"Found {self.config.NUM_CLASSES} classes: {class_names}")
        
        # Calculate dataset size
        total_train = train_generator.samples
        total_val = validation_generator.samples
        st.write(f"Training samples: {total_train}")
        st.write(f"Validation samples: {total_val}")
        
        return train_generator, validation_generator, class_names
    
    def visualize_samples(self, generator, class_names, num_samples=10):
        """Visualize random samples from the dataset"""
        # Get a batch of images
        images, labels = next(generator)
        
        # Create a Streamlit figure
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        for i in range(min(num_samples, len(images))):
            row, col = i // 5, i % 5
            # Convert preprocessed image back for visualization
            img = images[i]
            # Reverse the preprocessing if needed
            img = (img - img.min()) / (img.max() - img.min())
            axes[row, col].imshow(img)
            class_idx = np.argmax(labels[i])
            axes[row, col].set_title(f"Class: {class_names[class_idx]}")
            axes[row, col].axis('off')
        plt.tight_layout()
        return fig

# Model architecture
class EnhancedImageClassifier:
    def __init__(self, config):
        self.config = config
        
    def build_model(self):
    """Build an enhanced model with EfficientNetB3"""

    # Ensure NUM_CLASSES is set
    if self.config.NUM_CLASSES is None:
        raise ValueError("NUM_CLASSES is not set. Ensure dataset processing is completed before model building.")

    # Base pre-trained model
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.CHANNELS)
    )

    # Freeze the base model layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Create model
    inputs = Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.CHANNELS))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(self.config.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(self.config.DROPOUT_RATE)(x)

    # Error occurs here if NUM_CLASSES is None
    outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)  # Line 183

    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model
    
    def create_ensemble(self):
        """Create an ensemble of multiple models for better accuracy"""
        # First model: EfficientNetB3
        efficient_net_input = Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.CHANNELS))
        efficient_net = EfficientNetB3(
            weights='imagenet', 
            include_top=False,
            input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.CHANNELS)
        )(efficient_net_input)
        efficient_net = GlobalAveragePooling2D()(efficient_net)
        efficient_net = BatchNormalization()(efficient_net)
        efficient_net = Dense(512, activation='relu')(efficient_net)
        efficient_net = Dropout(0.3)(efficient_net)
        
        # Second model: ResNet50V2
        resnet_input = Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.CHANNELS))
        resnet = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.CHANNELS)
        )(resnet_input)
        resnet = GlobalAveragePooling2D()(resnet)
        resnet = BatchNormalization()(resnet)
        resnet = Dense(512, activation='relu')(resnet)
        resnet = Dropout(0.3)(resnet)
        
        # Concatenate models
        combined = tf.keras.layers.concatenate([efficient_net, resnet])
        
        # Add classification layers
        combined = Dense(512, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = BatchNormalization()(combined)
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        # Create ensemble model with multiple inputs
        ensemble_model = Model(inputs=[efficient_net_input, resnet_input], outputs=outputs)
        
        # Compile model
        ensemble_model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return ensemble_model
    
    def unfreeze_model(self, model, base_model):
        """Unfreeze base model for fine-tuning"""
        # Unfreeze the base model
        for layer in base_model.layers:
            layer.trainable = True
        
        # Recompile with a lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, train_generator, validation_generator):
        """Train the model with callbacks and Streamlit progress tracking"""
        # Create callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            # Save the best model
            ModelCheckpoint(
                filepath=self.config.MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Create Streamlit progress and metrics containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_col1, metrics_col2 = st.columns(2)
        train_acc_metric = metrics_col1.empty()
        val_acc_metric = metrics_col2.empty()
        
        # Custom callback for Streamlit updates
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Update progress bar
                progress = (epoch + 1) / config.EPOCHS
                progress_bar.progress(progress)
                
                # Update status text
                status_text.text(f"Epoch {epoch+1}/{config.EPOCHS} - Loss: {logs['loss']:.4f}")
                
                # Update metrics
                train_acc_metric.metric("Training Accuracy", f"{logs['accuracy']:.4f}")
                val_acc_metric.metric("Validation Accuracy", f"{logs['val_accuracy']:.4f}")
        
        # Add custom callback to callbacks list
        callbacks.append(StreamlitCallback())
        
        # Train the model
        st.write("Training in progress. Please wait...")
        history = model.fit(
            train_generator,
            epochs=self.config.EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        # Clear progress displays when done
        progress_bar.empty()
        status_text.empty()
        
        st.success("Training completed!")
        
        return model, history
    
    def evaluate_model(self, model, validation_generator):
        """Evaluate the model and visualize results"""
        # Get predictions
        validation_generator.reset()
        
        # Show a spinner during prediction
        with st.spinner("Generating predictions for evaluation..."):
            pred = model.predict(validation_generator)
            pred_classes = np.argmax(pred, axis=1)
            
            # Get true labels
            true_classes = validation_generator.classes
            class_names = list(validation_generator.class_indices.keys())
        
        # Classification report
        report = classification_report(true_classes, pred_classes, target_names=class_names, output_dict=True)
        
        # Create a DataFrame for better display in Streamlit
        report_df = pd.DataFrame(report).transpose()
        st.write("## Classification Report")
        st.dataframe(report_df.style.highlight_max(axis=0))
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        cm = confusion_matrix(true_classes, pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        st.write("## Confusion Matrix")
        st.pyplot(fig)
        
        return pred, true_classes

# Prediction and visualization
class Predictor:
    def __init__(self, model, config, class_names):
        self.model = model
        self.config = config
        self.class_names = class_names
    
    def predict_uploaded_image(self, uploaded_file):
        """Make prediction on an uploaded image file"""
        # Load and preprocess the image
        img = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize((self.config.IMG_SIZE, self.config.IMG_SIZE))
        st.image(img, caption="Uploaded Image", width=300)
        
        # Convert to array and preprocess
        img_array = np.array(img)
        img_preprocessed = efficientnet_preprocess(img_array)
        img_tensor = np.expand_dims(img_preprocessed, axis=0)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            prediction = self.model.predict(img_tensor)
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_predictions = [(self.class_names[i], prediction[0][i]) for i in top_indices]
        
        # Display results
        st.write("## Prediction Results")
        
        # Main prediction
        top_class_idx = top_indices[0]
        top_score = prediction[0][top_class_idx]
        predicted_class = self.class_names[top_class_idx]
        
        st.success(f"Predicted Class: **{predicted_class}** with confidence: **{top_score:.2%}**")
        
        # Create bar chart for top 3 predictions
        results_df = pd.DataFrame(
            [(cls, float(score*100)) for cls, score in top_predictions], 
            columns=['Class', 'Confidence (%)']
        )
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(results_df['Class'], results_df['Confidence (%)'], color='skyblue')
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Top 3 Predictions')
        
        # Add percentage labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                   ha='left', va='center')
        
        st.pyplot(fig)
        
        return predicted_class, top_score, top_predictions

# Batch Prediction
class BatchPredictor:
    def __init__(self, model, config, class_names):
        self.model = model
        self.config = config
        self.class_names = class_names
    
    def predict_batch(self, uploaded_files):
        """Process multiple uploaded files and make predictions"""
        results = []
        
        for file in uploaded_files:
            # Load and preprocess the image
            img = Image.open(file)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize((self.config.IMG_SIZE, self.config.IMG_SIZE))
            
            # Convert to array and preprocess
            img_array = np.array(img)
            img_preprocessed = efficientnet_preprocess(img_array)
            img_tensor = np.expand_dims(img_preprocessed, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img_tensor)
            
            # Get top prediction
            top_class_idx = np.argmax(prediction[0])
            top_score = prediction[0][top_class_idx]
            predicted_class = self.class_names[top_class_idx]
            
            results.append({
                'filename': file.name,
                'predicted_class': predicted_class,
                'confidence': top_score,
            })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        results_df['confidence'] = results_df['confidence'].apply(lambda x: f"{x:.2%}")
        
        return results_df

# Model Training and Evaluation Tab
def train_model_tab():
    st.header("Train Image Classification Model")
    st.write("Upload your dataset and train a custom image classification model.")
    
    # Dataset upload
    st.subheader("1. Upload Dataset")
    st.write("Please upload a zip file containing your dataset. The dataset should have a folder structure with class subfolders.")
    st.write("Example: dataset/class1/, dataset/class2/, etc.")
    
    uploaded_zip = st.file_uploader("Upload dataset zip file", type="zip")
    
    if uploaded_zip is not None:
        # Initialize dataset handler
        dataset_handler = DatasetHandler(config)
        
        # Process dataset
        if st.button("Process Dataset"):
            with st.spinner("Extracting and processing dataset..."):
                dataset_handler.process_uploaded_zip(uploaded_zip)
                
                # Process dataset
                train_generator, validation_generator, class_names = dataset_handler.process_dataset()
                
                # Store class names in session state
                st.session_state['class_names'] = class_names
                
                # Visualize sample images
                st.subheader("Dataset Samples:")
                fig = dataset_handler.visualize_samples(train_generator, class_names)
                st.pyplot(fig)
                
                # Store generators in session state
                st.session_state['train_generator'] = train_generator
                st.session_state['validation_generator'] = validation_generator
                
                # Set dataset processed flag
                st.session_state['dataset_processed'] = True
    
    # Model Selection and Training
    st.subheader("2. Model Selection and Training")
    
    if 'dataset_processed' in st.session_state and st.session_state['dataset_processed']:
        # Model choice
        model_type = st.radio(
            "Select model type:",
            ["Single Model (EfficientNetB3)", "Ensemble Model (EfficientNetB3 + ResNet50V2)"]
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        col1, col2, col3 = st.columns(3)
        config.BATCH_SIZE = col1.number_input("Batch Size", min_value=1, max_value=64, value=32)
        config.EPOCHS = col2.number_input("Epochs", min_value=1, max_value=100, value=30)
        config.LEARNING_RATE = col3.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
        
        # Start training button
        if st.button("Start Training"):
            # Build model
            classifier = EnhancedImageClassifier(config)
            
            if model_type == "Ensemble Model (EfficientNetB3 + ResNet50V2)":
                st.write("Building ensemble model (this will take longer but provides better accuracy)...")
                model = classifier.create_ensemble()
                # Store model in session state
                st.session_state['model'] = model
                st.session_state['model_type'] = 'ensemble'
            else:
                model, base_model = classifier.build_model()
                # Store model and base_model in session state
                st.session_state['model'] = model
                st.session_state['base_model'] = base_model
                st.session_state['model_type'] = 'single'
            
            # Train model
            model, history = classifier.train_model(
                st.session_state['model'], 
                st.session_state['train_generator'], 
                st.session_state['validation_generator']
            )
            
            # Store trained model in session state
            st.session_state['trained_model'] = model
            st.session_state['training_history'] = history
            
            # Plot training history
            st.subheader("Training History")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Accuracy plot
            ax1.plot(history.history['accuracy'])
            ax1.plot(history.history['val_accuracy'])
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend(['Train', 'Validation'], loc='lower right')
            
            # Loss plot
            ax2.plot(history.history['loss'])
            ax2.plot(history.history['val_loss'])
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend(['Train', 'Validation'], loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Evaluate model
            st.subheader("Model Evaluation")
            predictions, true_labels = classifier.evaluate_model(
                model, 
                st.session_state['validation_generator']
            )
            
            # Save model button
            if st.button("Save Trained Model"):
                model.save(config.MODEL_PATH)
                # Provide download link
                with open(config.MODEL_PATH, "rb") as file:
                    btn = st.download_button(
                        label="Download Model",
                        data=file,
                        file_name="image_classification_model.h5",
                        mime="application/octet-stream"
                    )
                st.success(f"Model saved as {config.MODEL_PATH}")
    else:
        st.info("Please upload and process your dataset first.")

# Prediction Tab
def prediction_tab():
    st.header("Image Prediction")
    
    # Model upload or use trained model
    st.subheader("1. Select Model")
    model_option = st.radio(
        "Choose model source:",
        ["Use trained model from this session", "Upload a saved model"]
    )
    
    # Initialize with None
    model = None
    class_names = None
    
    if model_option == "Use trained model from this session":
        if 'trained_model' in st.session_state and 'class_names' in st.session_state:
            model = st.session_state['trained_model']
            class_names = st.session_state['class_names']
            st.success("Using model trained in this session")
        else:
            st.warning("No trained model found in this session. Please train a model first or upload a saved model.")
    else:
        uploaded_model = st.file_uploader("Upload model file (.h5)", type="h5")
        
        if uploaded_model is not None:
            # Save the uploaded model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(uploaded_model.getbuffer())
                model_path = tmp_file.name
            
            # Load the model
            model = load_model(model_path)
            st.success("Model loaded successfully")
            
            # Ask for class names
            class_names_input = st.text_input(
                "Enter class names (comma separated)", 
                help="Example: cat,dog,horse"
            )
            
            if class_names_input:
                class_names = [c.strip() for c in class_names_input.split(',')]
                st.write(f"Classes: {class_names}")
    
    # Image prediction
    if model is not None and class_names is not None:
        st.subheader("2. Upload Image for Prediction")
        
        # Create config for predictor
        pred_config = Config()
        pred_config.NUM_CLASSES = len(class_names)
        
        # Create predictor
        predictor = Predictor(model, pred_config, class_names)
        
        # Prediction mode
        prediction_mode = st.radio(
            "Choose prediction mode:",
            ["Single Image", "Batch Processing"]
        )
        
        if prediction_mode == "Single Image":
            # Upload single image
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Predict and show results
                predicted_class, confidence, top3 = predictor.predict_uploaded_image(uploaded_file)
                
                # Additional details in expander
                with st.expander("See Technical Details"):
                    st.write("**Model Architecture:**")
                    st.text(model.summary())
        else:
            # Upload multiple images for batch processing
            st.write("Upload multiple images for batch prediction")
            batch_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            
            if batch_files and len(batch_files) > 0:
                batch_predictor = BatchPredictor(model, pred_config, class_names)
                
                with st.spinner(f"Processing {len(batch_files)} images..."):
                    results_df = batch_predictor.predict_batch(batch_files)
                
                # Display results table
                st.subheader("Batch Prediction Results")
                st.dataframe(results_df)
                
                # Export results to CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv",
                )
    else:
        st.info("Please select or upload a model first")

# About Tab
def about_tab():
    st.header("Advanced Image Classification")
    st.write("""
    This application allows you to train and deploy state-of-the-art image classification models.
    It's built for the Infosys Hackathon and includes features for both model training and inference.
    """)
    
    st.subheader("Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Training:**")
        st.markdown("""
        - Works with full custom datasets
        - High-accuracy model architecture
        - Advanced data augmentation
        - Multiple model ensemble option
        - Visualization tools
        """)
    
    with col2:
        st.write("**Prediction:**")
        st.markdown("""
        - Single image prediction
        - Batch processing support
        - Confidence visualization
        - Export results to CSV
        - Pre-trained model upload
        """)
    
    st.subheader("How to Use")
    st.markdown("""
    1. **Train a New Model:**
       - Go to the 'Train Model' tab
       - Upload your dataset as a zip file (with class subfolders)
       - Configure and train your model
       - Evaluate and download the trained model
       
    2. **Make Predictions:**
       - Go to the 'Prediction' tab
       - Use your trained model or upload a saved model
       - Upload images for classification
       - View results and confidence scores
    """)

# Main Streamlit app
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .subheader {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #2563EB;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<p class="main-header">Advanced Image Classification System</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'dataset_processed' not in st.session_state:
        st.session_state['dataset_processed'] = False
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Train Model", "Prediction", "About"])
    
    with tab1:
        train_model_tab()
    
    with tab2:
        prediction_tab()
    
    with tab3:
        about_tab()

# Run the app
if __name__ == "__main__":
    main()
