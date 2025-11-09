
# Create training script

training_script = """
# Training Script for Air Quality Detection Model
# Train the CNN-LSTM model on air quality datasets

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from air_quality_detector import AirQualityDetector
import pandas as pd
import cv2
from tqdm import tqdm

class AirQualityTrainer:
    '''
    Training pipeline for air quality detection model
    '''
    
    def __init__(self, data_dir, sequence_length=5):
        '''
        Initialize trainer
        Args:
            data_dir: Directory containing training data
            sequence_length: Number of images in sequence
        '''
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.detector = AirQualityDetector()
        self.image_size = (224, 224)
        
    def load_dataset(self, csv_path=None):
        '''
        Load dataset from directory structure or CSV
        Expected structure:
        data_dir/
            Good/
                img1.jpg
                img2.jpg
            Moderate/
                img1.jpg
            ...
        
        Or CSV with columns: image_path, aqi_category
        '''
        X_sequences = []
        y_labels = []
        
        if csv_path and os.path.exists(csv_path):
            # Load from CSV
            df = pd.read_csv(csv_path)
            print(f"Loading {len(df)} images from CSV...")
            
            # Group by sequence_id if available
            if 'sequence_id' in df.columns:
                for seq_id, group in df.groupby('sequence_id'):
                    if len(group) >= self.sequence_length:
                        images = []
                        for idx, row in group.head(self.sequence_length).iterrows():
                            img = self.load_and_preprocess(row['image_path'])
                            if img is not None:
                                images.append(img)
                        
                        if len(images) == self.sequence_length:
                            X_sequences.append(np.array(images))
                            y_labels.append(row['aqi_category'])
            else:
                # Single image mode - create sequences by repetition
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    img = self.load_and_preprocess(row['image_path'])
                    if img is not None:
                        # Repeat image to create sequence
                        sequence = np.repeat(img[np.newaxis, ...], 
                                           self.sequence_length, axis=0)
                        X_sequences.append(sequence)
                        y_labels.append(row['aqi_category'])
        
        else:
            # Load from directory structure
            print(f"Loading images from directory: {self.data_dir}")
            categories = sorted(os.listdir(self.data_dir))
            
            for category in categories:
                category_path = os.path.join(self.data_dir, category)
                if not os.path.isdir(category_path):
                    continue
                
                images_in_category = sorted([
                    f for f in os.listdir(category_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                
                print(f"Category: {category} - {len(images_in_category)} images")
                
                # Create sequences from images
                for i in range(0, len(images_in_category), self.sequence_length):
                    sequence_images = images_in_category[i:i+self.sequence_length]
                    
                    if len(sequence_images) == self.sequence_length:
                        images = []
                        for img_name in sequence_images:
                            img_path = os.path.join(category_path, img_name)
                            img = self.load_and_preprocess(img_path)
                            if img is not None:
                                images.append(img)
                        
                        if len(images) == self.sequence_length:
                            X_sequences.append(np.array(images))
                            y_labels.append(category)
        
        X = np.array(X_sequences)
        y = np.array(y_labels)
        
        print(f"\\nLoaded {len(X)} sequences")
        print(f"Sequence shape: {X.shape}")
        print(f"Unique categories: {np.unique(y)}")
        
        return X, y
    
    def load_and_preprocess(self, image_path):
        '''
        Load and preprocess single image
        '''
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img = cv2.resize(img, self.image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.1):
        '''
        Prepare training, validation, and test sets
        '''
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        
        print(f"\\nData split:")
        print(f"Training: {len(X_train)} sequences")
        print(f"Validation: {len(X_val)} sequences")
        print(f"Test: {len(X_test)} sequences")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder
    
    def create_data_augmentation(self):
        '''
        Create data augmentation pipeline
        '''
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        return augmentation
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=8):
        '''
        Train the air quality detection model
        '''
        # Build model
        num_classes = y_train.shape[1]
        print(f"\\nBuilding model for {num_classes} classes...")
        self.detector.build_cnn_lstm_model(num_classes=num_classes)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1
            )
        ]
        
        # Train model
        print("\\nStarting training...")
        history = self.detector.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        '''
        Evaluate model on test set
        '''
        print("\\nEvaluating model on test set...")
        test_loss, test_acc, test_top_k = self.detector.model.evaluate(
            X_test, y_test, verbose=1
        )
        
        print(f"\\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-K Accuracy: {test_top_k:.4f}")
        
        return test_loss, test_acc, test_top_k
    
    def save_training_results(self, history, test_results, label_encoder):
        '''
        Save training history and results
        '''
        os.makedirs('results', exist_ok=True)
        
        # Save history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('results/training_history.csv', index=False)
        
        # Save test results
        results = {
            'test_loss': test_results[0],
            'test_accuracy': test_results[1],
            'test_top_k_accuracy': test_results[2]
        }
        
        with open('results/test_results.txt', 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\\n")
        
        # Save label encoder
        np.save('models/label_encoder_classes.npy', label_encoder.classes_)
        
        print("\\nResults saved to 'results/' directory")

# Main training pipeline
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/air_quality_images"
    CSV_PATH = "data/air_quality_data.csv"  # Optional
    EPOCHS = 50
    BATCH_SIZE = 8
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize trainer
    print("="*60)
    print("Air Quality Detection Model Training")
    print("="*60)
    
    trainer = AirQualityTrainer(DATA_DIR, sequence_length=5)
    
    # Load dataset
    if os.path.exists(CSV_PATH):
        X, y = trainer.load_dataset(csv_path=CSV_PATH)
    else:
        X, y = trainer.load_dataset()
    
    # Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = trainer.prepare_data(X, y)
    
    # Train model
    history = trainer.train_model(X_train, y_train, X_val, y_val, 
                                   epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Evaluate model
    test_results = trainer.evaluate_model(X_test, y_test)
    
    # Save results
    trainer.save_training_results(history, test_results, label_encoder)
    
    # Save final model
    trainer.detector.save_model('models/air_quality_model_final.h5')
    
    print("\\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
"""

with open('train_model.py', 'w') as f:
    f.write(training_script)

print("Training script saved: train_model.py")
