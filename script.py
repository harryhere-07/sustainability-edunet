
# Create the main project structure with Python code for air quality assessment

main_code = """
# Real-Time Urban Air Quality Assessment and Particulate Matter Detection
# Using Computer Vision and Deep Learning

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import json

class AirQualityDetector:
    '''
    Main class for air quality detection using computer vision and deep learning
    '''
    
    def __init__(self, model_path=None):
        '''
        Initialize the air quality detector
        Args:
            model_path: Path to pre-trained model (optional)
        '''
        self.model = None
        self.image_size = (224, 224)
        self.aqi_categories = {
            0: 'Good (0-50)',
            1: 'Moderate (51-100)',
            2: 'Unhealthy for Sensitive Groups (101-150)',
            3: 'Unhealthy (151-200)',
            4: 'Very Unhealthy (201-300)',
            5: 'Hazardous (>300)'
        }
        
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("No model loaded. Please build or load a model.")
    
    def build_cnn_lstm_model(self, num_classes=6):
        '''
        Build CNN-LSTM hybrid model for air quality estimation
        Returns:
            Compiled Keras model
        '''
        # Input layer for image sequences
        input_layer = layers.Input(shape=(5, *self.image_size, 3))
        
        # TimeDistributed CNN for spatial feature extraction
        base_model = ResNet50(weights='imagenet', include_top=False, 
                             input_shape=(*self.image_size, 3))
        
        # Freeze early layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Apply CNN to each timestep
        x = layers.TimeDistributed(base_model)(input_layer)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # LSTM for temporal feature extraction
        x = layers.LSTM(256, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers for classification
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output_layer = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def extract_haze_features(self, image):
        '''
        Extract haze-related features from image using computer vision
        Args:
            image: Input image (BGR format)
        Returns:
            Dictionary of haze features
        '''
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Dark Channel Prior
        dark_channel = self.compute_dark_channel(image)
        
        # 2. Contrast and Visibility
        contrast = np.std(gray)
        mean_intensity = np.mean(gray)
        
        # 3. Color Attenuation
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        color_attenuation = np.mean(value) - np.mean(saturation)
        
        # 4. Edge Density (for visibility estimation)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 5. Haze Density Estimation
        haze_density = np.mean(dark_channel) / 255.0
        
        features = {
            'dark_channel_mean': np.mean(dark_channel),
            'contrast': contrast,
            'mean_intensity': mean_intensity,
            'color_attenuation': color_attenuation,
            'edge_density': edge_density,
            'haze_density': haze_density,
            'saturation_mean': np.mean(saturation),
            'visibility_score': self.estimate_visibility(edge_density, contrast)
        }
        
        return features
    
    def compute_dark_channel(self, image, patch_size=15):
        '''
        Compute dark channel prior for haze estimation
        Args:
            image: Input BGR image
            patch_size: Size of local patch
        Returns:
            Dark channel image
        '''
        b, g, r = cv2.split(image)
        min_channel = cv2.min(cv2.min(r, g), b)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def estimate_atmospheric_light(self, image, dark_channel):
        '''
        Estimate atmospheric light using dark channel prior
        Args:
            image: Input BGR image
            dark_channel: Computed dark channel
        Returns:
            Atmospheric light value
        '''
        h, w = dark_channel.shape
        num_pixels = int(h * w * 0.001)  # Top 0.1% brightest pixels
        
        flat_dark = dark_channel.flatten()
        flat_image = image.reshape(-1, 3)
        
        indices = np.argsort(flat_dark)[-num_pixels:]
        atmospheric_light = np.max(flat_image[indices], axis=0)
        
        return atmospheric_light
    
    def estimate_transmission(self, image, atmospheric_light, omega=0.95, patch_size=15):
        '''
        Estimate transmission map for dehazing
        Args:
            image: Input BGR image
            atmospheric_light: Estimated atmospheric light
            omega: Haze retention parameter
            patch_size: Size of local patch
        Returns:
            Transmission map
        '''
        normalized = image.astype(np.float64) / atmospheric_light
        normalized = np.clip(normalized, 0, 1)
        
        transmission = 1 - omega * self.compute_dark_channel(
            (normalized * 255).astype(np.uint8), patch_size
        ) / 255.0
        
        return transmission
    
    def estimate_visibility(self, edge_density, contrast, max_visibility=10.0):
        '''
        Estimate visibility range in kilometers
        Args:
            edge_density: Density of edges in image
            contrast: Image contrast
            max_visibility: Maximum visibility range
        Returns:
            Estimated visibility in km
        '''
        # Visibility is positively correlated with edge density and contrast
        visibility = (edge_density * 5 + contrast / 50) * max_visibility
        visibility = np.clip(visibility, 0.1, max_visibility)
        
        return visibility
    
    def preprocess_image(self, image_path):
        '''
        Preprocess image for model inference
        Args:
            image_path: Path to image file
        Returns:
            Preprocessed image array
        '''
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, self.image_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb / 255.0
        
        return img_normalized
    
    def predict_air_quality(self, image_path):
        '''
        Predict air quality from single image
        Args:
            image_path: Path to image file
        Returns:
            Dictionary with prediction results
        '''
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Read and preprocess image
        img = cv2.imread(image_path)
        preprocessed = self.preprocess_image(image_path)
        
        # Extract haze features
        haze_features = self.extract_haze_features(img)
        
        # For single image prediction, repeat to create sequence
        img_sequence = np.expand_dims(np.repeat(
            preprocessed[np.newaxis, ...], 5, axis=0), axis=0)
        
        # Make prediction
        prediction = self.model.predict(img_sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        result = {
            'predicted_aqi_category': self.aqi_categories[predicted_class],
            'confidence': float(confidence),
            'class_probabilities': {
                self.aqi_categories[i]: float(prediction[0][i])
                for i in range(len(self.aqi_categories))
            },
            'haze_features': haze_features,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def predict_sequence(self, image_paths):
        '''
        Predict air quality from sequence of images
        Args:
            image_paths: List of 5 consecutive image paths
        Returns:
            Dictionary with prediction results
        '''
        if self.model is None:
            return {"error": "Model not loaded"}
        
        if len(image_paths) != 5:
            return {"error": "Exactly 5 images required for sequence prediction"}
        
        # Preprocess all images
        sequence = np.array([self.preprocess_image(path) for path in image_paths])
        sequence = np.expand_dims(sequence, axis=0)
        
        # Make prediction
        prediction = self.model.predict(sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        result = {
            'predicted_aqi_category': self.aqi_categories[predicted_class],
            'confidence': float(confidence),
            'class_probabilities': {
                self.aqi_categories[i]: float(prediction[0][i])
                for i in range(len(self.aqi_categories))
            },
            'num_images_in_sequence': len(image_paths),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def visualize_prediction(self, image_path, save_path=None):
        '''
        Visualize prediction with haze analysis
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
        '''
        # Read image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        result = self.predict_air_quality(image_path)
        
        # Extract features
        dark_channel = self.compute_dark_channel(img)
        atm_light = self.estimate_atmospheric_light(img, dark_channel)
        transmission = self.estimate_transmission(img, atm_light)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Dark channel
        axes[0, 1].imshow(dark_channel, cmap='gray')
        axes[0, 1].set_title('Dark Channel Prior')
        axes[0, 1].axis('off')
        
        # Transmission map
        axes[0, 2].imshow(transmission, cmap='jet')
        axes[0, 2].set_title('Transmission Map')
        axes[0, 2].axis('off')
        
        # Edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection (Visibility)')
        axes[1, 0].axis('off')
        
        # Prediction results
        axes[1, 1].axis('off')
        prediction_text = f"Predicted AQI: {result['predicted_aqi_category']}\\n"
        prediction_text += f"Confidence: {result['confidence']:.2%}\\n\\n"
        prediction_text += "Haze Features:\\n"
        for key, value in result['haze_features'].items():
            prediction_text += f"{key}: {value:.3f}\\n"
        
        axes[1, 1].text(0.1, 0.5, prediction_text, fontsize=10, 
                       verticalalignment='center', family='monospace')
        axes[1, 1].set_title('Prediction & Features')
        
        # Class probabilities
        categories = list(result['class_probabilities'].keys())
        probabilities = list(result['class_probabilities'].values())
        axes[1, 2].barh(categories, probabilities)
        axes[1, 2].set_xlabel('Probability')
        axes[1, 2].set_title('AQI Category Probabilities')
        axes[1, 2].set_xlim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, path):
        '''Save the model to disk'''
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
    
    def get_model_summary(self):
        '''Print model architecture summary'''
        if self.model:
            return self.model.summary()
        else:
            print("No model loaded")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = AirQualityDetector()
    
    # Build model
    print("Building CNN-LSTM model...")
    model = detector.build_cnn_lstm_model(num_classes=6)
    print("\\nModel architecture:")
    model.summary()
    
    print("\\n" + "="*50)
    print("Air Quality Detector initialized successfully!")
    print("Model is ready for training or inference.")
    print("="*50)
"""

# Save the main code
with open('air_quality_detector.py', 'w') as f:
    f.write(main_code)

print("Main code saved: air_quality_detector.py")
print("=" * 60)
