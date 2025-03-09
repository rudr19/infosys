import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, MultiHeadAttention
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, Xception
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
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
import math
import cv2

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

class DatasetHandler:
    def __init__(self, config):
        self.config = config

    def process_uploaded_zip(self, uploaded_zip):
        """Extracts uploaded dataset ZIP file and saves it to the dataset path."""
        dataset_path = self.config.DATASET_PATH  

        # Ensure dataset directory exists
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # Save ZIP file temporarily
        temp_zip_path = os.path.join(dataset_path, "dataset.zip")
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        # Extract ZIP file
        try:
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(dataset_path)
            os.remove(temp_zip_path)  # Remove ZIP after extraction
            st.success("Dataset uploaded and extracted successfully!")
        except Exception as e:
            st.error(f"Error extracting dataset: {str(e)}")
            return False

        return True

    def process_dataset(self):
        """Process datasets in different formats"""
        print(f"DEBUG: Checking dataset at {self.config.DATASET_PATH}")

        if not os.path.exists(self.config.DATASET_PATH):
            st.error(f"Dataset path {self.config.DATASET_PATH} not found!")
            return None, None, None

        # Detect dataset format
        dataset_format = self.detect_dataset_format()
        
        if dataset_format == "folders":
            return self.process_folder_structure()
        elif dataset_format == "csv":
            return self.process_csv_labels()
        elif dataset_format == "single_folder":
            return self.process_single_folder()
        else:
            st.error("Unsupported dataset format! Ensure images have valid class labels.")
            return None, None, None

    def detect_dataset_format(self):
        """Detects the dataset format"""
        dataset_path = self.config.DATASET_PATH

        if any(os.path.isdir(os.path.join(dataset_path, d)) for d in os.listdir(dataset_path)):
            return "folders"

        if any(f.endswith(".csv") for f in os.listdir(dataset_path)):
            return "csv"

        if all(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(dataset_path)):
            return "single_folder"

        return None

    def process_folder_structure(self):
        """Process dataset with standard folder-based structure"""
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

        try:
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

            class_names = list(train_generator.class_indices.keys())
            self.config.NUM_CLASSES = len(class_names)

            print(f"DEBUG: Found classes {class_names}")
            print(f"DEBUG: NUM_CLASSES set to {self.config.NUM_CLASSES}")

            if self.config.NUM_CLASSES is None or self.config.NUM_CLASSES <= 0:
                st.error("Dataset processing failed. No valid class folders found.")
                return None, None, None

            return train_generator, validation_generator, class_names

        except Exception as e:
            st.error(f"Error during dataset processing: {str(e)}")
            print(f"ERROR: {str(e)}")
            return None, None, None

    def process_csv_labels(self):
        """Process dataset using a CSV file with image paths and labels"""
        dataset_path = self.config.DATASET_PATH
        csv_file = next((f for f in os.listdir(dataset_path) if f.endswith(".csv")), None)

        if not csv_file:
            st.error("No CSV file found in dataset directory.")
            return None, None, None

        df = pd.read_csv(os.path.join(dataset_path, csv_file))

        if "image_path" not in df.columns or "label" not in df.columns:
            st.error("CSV must contain 'image_path' and 'label' columns.")
            return None, None, None

        class_names = df["label"].unique().tolist()
        self.config.NUM_CLASSES = len(class_names)

        images = []
        labels = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(dataset_path, row["image_path"])
            if os.path.exists(img_path):
                img = load_img(img_path, target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_names.index(row["label"]))

        images = np.array(images) / 255.0
        labels = np.array(labels)

        return images, labels, class_names

    def process_single_folder(self):
        """Process a dataset with all images in a single folder (No subfolders)"""
        dataset_path = self.config.DATASET_PATH
        image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not image_files:
            st.error("No valid image files found in the dataset.")
            return None, None, None

        images = []
        labels = []
        
        for img_file in image_files:
            img_path = os.path.join(dataset_path, img_file)
            img = load_img(img_path, target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(0)  # No specific class, single-folder dataset

        images = np.array(images) / 255.0
        labels = np.array(labels)

        self.config.NUM_CLASSES = 1  # Single-class dataset

        return images, labels, ["SingleClass"]

    def visualize_samples(self, generator, class_names, num_samples=10):
        """Visualize random samples from the dataset"""
        try:
            images, labels = next(generator)

            fig, axes = plt.subplots(2, 5, figsize=(20, 10))
            for i in range(min(num_samples, len(images))):
                row, col = i // 5, i % 5
                img = images[i]
                img = (img - img.min()) / (img.max() - img.min())  # Normalize for visualization
                axes[row, col].imshow(img)
                class_idx = np.argmax(labels[i])
                axes[row, col].set_title(f"Class: {class_names[class_idx]}")
                axes[row, col].axis('off')
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error while visualizing samples: {str(e)}")
            return None

# Model architecture (Updated with Enhancements)
class EnhancedImageClassifier:
    def __init__(self, config):
        self.config = config

    def build_model(self):
        """Build an enhanced model with EfficientNetB3 and attention mechanism"""
        if self.config.NUM_CLASSES is None or self.config.NUM_CLASSES <= 0:
            raise ValueError("ERROR: NUM_CLASSES is not set or invalid. Ensure dataset processing is completed before model building.")

        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.CHANNELS)
        )

        for layer in base_model.layers:
            layer.trainable = False

        inputs = Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, self.config.CHANNELS))
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = tf.expand_dims(x, axis=1)  # Add sequence dimension for attention
        x = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)  # Attention layer
        x = tf.squeeze(x, axis=1)  # Remove sequence dimension
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)

        if self.config.NUM_CLASSES == 1:
            outputs = Dense(1, activation='sigmoid')(x)
            loss_function = 'binary_crossentropy'
        else:
            outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)
            loss_function = 'categorical_crossentropy'

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss=loss_function,
            metrics=['accuracy']
        )

        return model, base_model
    
    def create_ensemble(self):
        """Create an ensemble of multiple models for better accuracy"""
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
        
        combined = tf.keras.layers.concatenate([efficient_net, resnet])
        
        combined = Dense(512, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = BatchNormalization()(combined)
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(combined)
        
        ensemble_model = Model(inputs=[efficient_net_input, resnet_input], outputs=outputs)
        
        ensemble_model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return ensemble_model
    
    def unfreeze_model(self, model, base_model):
        """Unfreeze base model for fine-tuning"""
        for layer in base_model.layers:
            layer.trainable = True
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, train_generator, validation_generator):
        """Train the model with mixed precision, cosine annealing, and quantization"""
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('mixed_float16')  # Enable mixed precision training

        # Cosine Annealing Scheduler
        class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
            def __init__(self, max_lr, min_lr, epochs):
                super().__init__()
                self.max_lr = max_lr
                self.min_lr = min_lr
                self.epochs = epochs

            def on_epoch_begin(self, epoch, logs=None):
                lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / self.epochs)) / 2
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint(filepath=self.config.MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
            CosineAnnealingScheduler(max_lr=self.config.LEARNING_RATE, min_lr=1e-6, epochs=self.config.EPOCHS)
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_col1, metrics_col2 = st.columns(2)
        train_acc_metric = metrics_col1.empty()
        val_acc_metric = metrics_col2.empty()
        
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / config.EPOCHS
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{config.EPOCHS} - Loss: {logs['loss']:.4f}")
                train_acc_metric.metric("Training Accuracy", f"{logs['accuracy']:.4f}")
                val_acc_metric.metric("Validation Accuracy", f"{logs['val_accuracy']:.4f}")
        
        callbacks.append(StreamlitCallback())
        
        st.write("Training in progress. Please wait...")
        history = model.fit(
            train_generator,
            epochs=self.config.EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        progress_bar.empty()
        status_text.empty()
        st.success("Training completed!")
        
        # Quantize model for deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        st.download_button("Download Quantized Model", tflite_model, "model.tflite", "application/octet-stream")
        
        return model, history
    
    def evaluate_model(self, model, validation_generator):
        """Evaluate the model and visualize results"""
        validation_generator.reset()
        
        with st.spinner("Generating predictions for evaluation..."):
            pred = model.predict(validation_generator)
            pred_classes = np.argmax(pred, axis=1)
            true_classes = validation_generator.classes
            class_names = list(validation_generator.class_indices.keys())
        
        report = classification_report(true_classes, pred_classes, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("## Classification Report")
        st.dataframe(report_df.style.highlight_max(axis=0))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        cm = confusion_matrix(true_classes, pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        st.write("## Confusion Matrix")
        st.pyplot(fig)
        
        return pred, true_classes

# Prediction and visualization (Updated with Enhancements)
class Predictor:
    def __init__(self, model, config, class_names):
        self.model = model
        self.config = config
        self.class_names = class_names
    
    def get_gradcam_heatmap(self, img_tensor, pred_class_idx):
        """Generate Grad-CAM heatmap for explainability"""
        last_conv_layer = [layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
        grad_model = Model(self.model.inputs, [last_conv_layer.output, self.model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, pred_class_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    def predict_uploaded_image(self, uploaded_file):
        """Make prediction with uncertainty estimation and Grad-CAM"""
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((self.config.IMG_SIZE, self.config.IMG_SIZE))
        st.image(img, caption="Uploaded Image", width=300)
        
        img_array = np.array(img)
        img_preprocessed = efficientnet_preprocess(img_array)
        img_tensor = np.expand_dims(img_preprocessed, axis=0)
        
        # Monte Carlo Dropout for uncertainty
        with st.spinner("Analyzing image with uncertainty estimation..."):
            predictions = [self.model(img_tensor, training=True) for _ in range(10)]  # 10 samples
            mean_pred = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
        
        top_indices = np.argsort(mean_pred[0])[-3:][::-1]
        top_predictions = [(self.class_names[i], mean_pred[0][i], uncertainty[i]) for i in top_indices]
        
        top_class_idx = top_indices[0]
        top_score = mean_pred[0][top_class_idx]
        predicted_class = self.class_names[top_class_idx]
        
        # Grad-CAM visualization
        heatmap = self.get_gradcam_heatmap(img_tensor, top_class_idx)
        heatmap = cv2.resize(heatmap, (self.config.IMG_SIZE, self.config.IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        st.image(heatmap, caption="Grad-CAM Heatmap", width=300)
        
        st.success(f"Predicted Class: **{predicted_class}** with confidence: **{top_score:.2%}** (± {uncertainty[top_class_idx]:.2%})")
        
        results_df = pd.DataFrame(
            [(cls, float(score*100), float(unc*100)) for cls, score, unc in top_predictions], 
            columns=['Class', 'Confidence (%)', 'Uncertainty (±%)']
        )
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(results_df['Class'], results_df['Confidence (%)'], color='skyblue')
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Top 3 Predictions with Uncertainty')
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}% (±{results_df["Uncertainty (±%)"].iloc[i]:.1f}%)', 
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
            img = Image.open(file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((self.config.IMG_SIZE, self.config.IMG_SIZE))
            img_array = np.array(img)
            img_preprocessed = efficientnet_preprocess(img_array)
            img_tensor = np.expand_dims(img_preprocessed, axis=0)
            prediction = self.model.predict(img_tensor)
            top_class_idx = np.argmax(prediction[0])
            top_score = prediction[0][top_class_idx]
            predicted_class = self.class_names[top_class_idx]
            results.append({
                'filename': file.name,
                'predicted_class': predicted_class,
                'confidence': top_score,
            })
        
        results_df = pd.DataFrame(results)
        results_df['confidence'] = results_df['confidence'].apply(lambda x: f"{x:.2%}")
        return results_df
        
# Model Training and Evaluation Tab
def train_model_tab():
    st.header("Train Image Classification Model")
    st.write("Upload your dataset and train a custom image classification model.")
    
    st.subheader("1. Upload Dataset")
    st.write("Please upload a zip file containing your dataset. The dataset should have a folder structure with class subfolders.")
    st.write("Example: dataset/class1/, dataset/class2/, etc.")
    
    uploaded_zip = st.file_uploader("Upload dataset zip file", type="zip")
    
    if uploaded_zip is not None:
        dataset_handler = DatasetHandler(config)  
        train_generator, validation_generator, class_names = dataset_handler.process_dataset()
        
        if st.button("Process Dataset"):
            with st.spinner("Extracting and processing dataset..."):
                dataset_handler = DatasetHandler(config)
                dataset_handler.process_uploaded_zip(uploaded_zip)
                train_generator, validation_generator, class_names = dataset_handler.process_dataset()
                
                if config.NUM_CLASSES is None or config.NUM_CLASSES <= 0:
                    st.error("Dataset processing failed. No valid classes were found.")
                    return
                
                st.session_state['class_names'] = class_names
                st.subheader("Dataset Samples:")
                fig = dataset_handler.visualize_samples(train_generator, class_names)
                st.pyplot(fig)
                st.session_state['train_generator'] = train_generator
                st.session_state['validation_generator'] = validation_generator
                st.session_state['dataset_processed'] = True

    st.subheader("2. Model Selection and Training")
    
    if 'dataset_processed' in st.session_state and st.session_state['dataset_processed']:
        st.write(f"DEBUG: NUM_CLASSES before building model = {config.NUM_CLASSES}")

        if config.NUM_CLASSES is None or config.NUM_CLASSES <= 0:
            st.error("Dataset processing failed. NUM_CLASSES is not valid. Please reprocess the dataset.")
            return
        
        model_type = st.radio(
            "Select model type:",
            ["Single Model (EfficientNetB3)", "Ensemble Model (EfficientNetB3 + ResNet50V2)"]
        )
        
        st.subheader("Training Parameters")
        col1, col2, col3 = st.columns(3)
        config.BATCH_SIZE = col1.number_input("Batch Size", min_value=1, max_value=64, value=32)
        config.EPOCHS = col2.number_input("Epochs", min_value=1, max_value=100, value=30)
        config.LEARNING_RATE = col3.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
        
        if st.button("Start Training"):
            classifier = EnhancedImageClassifier(config)
            
            if model_type == "Ensemble Model (EfficientNetB3 + ResNet50V2)":
                st.write("Building ensemble model (this will take longer but provides better accuracy)...")
                model = classifier.create_ensemble()
            else:
                model, base_model = classifier.build_model()
            
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type
            
            model, history = classifier.train_model(
                st.session_state['model'], 
                st.session_state['train_generator'], 
                st.session_state['validation_generator']
            )
            
            st.session_state['trained_model'] = model
            st.session_state['training_history'] = history
            
            st.subheader("Training History")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(history.history['accuracy'])
            ax1.plot(history.history['val_accuracy'])
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend(['Train', 'Validation'], loc='lower right')
            ax2.plot(history.history['loss'])
            ax2.plot(history.history['val_loss'])
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend(['Train', 'Validation'], loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Model Evaluation")
            predictions, true_labels = classifier.evaluate_model(
                model, 
                st.session_state['validation_generator']
            )
            
            if st.button("Save Trained Model"):
                model.save(config.MODEL_PATH)
                with open(config.MODEL_PATH, "rb") as file:
                    st.download_button(
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
    
    st.subheader("1. Select Model")
    model_option = st.radio(
        "Choose model source:",
        ["Use trained model from this session", "Upload a saved model"]
    )
    
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
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(uploaded_model.getbuffer())
                model_path = tmp_file.name
            
            model = load_model(model_path)
            st.success("Model loaded successfully")
            
            class_names_input = st.text_input(
                "Enter class names (comma separated)", 
                help="Example: cat,dog,horse"
            )
            
            if class_names_input:
                class_names = [c.strip() for c in class_names_input.split(',')]
                st.write(f"Classes: {class_names}")
    
    if model is not None and class_names is not None:
        st.subheader("2. Upload Image for Prediction")
        
        pred_config = Config()
        pred_config.NUM_CLASSES = len(class_names)
        
        predictor = Predictor(model, pred_config, class_names)
        
        prediction_mode = st.radio(
            "Choose prediction mode:",
            ["Single Image", "Batch Processing"]
        )
        
        if prediction_mode == "Single Image":
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                predicted_class, confidence, top3 = predictor.predict_uploaded_image(uploaded_file)
                
                with st.expander("See Technical Details"):
                    st.write("**Model Architecture:**")
                    st.text(model.summary())
        else:
            st.write("Upload multiple images for batch prediction")
            batch_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            
            if batch_files and len(batch_files) > 0:
                batch_predictor = BatchPredictor(model, pred_config, class_names)
                
                with st.spinner(f"Processing {len(batch_files)} images..."):
                    results_df = batch_predictor.predict_batch(batch_files)
                
                st.subheader("Batch Prediction Results")
                st.dataframe(results_df)
                
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
    
    st.markdown('<p class="main-header">Advanced Image Classification System</p>', unsafe_allow_html=True)
    
    if 'dataset_processed' not in st.session_state:
        st.session_state['dataset_processed'] = False
    
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
