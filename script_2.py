
# Create inference script for real-time prediction

inference_script = """
# Real-Time Inference Script for Air Quality Detection
# Use trained model to predict air quality from images or video

import os
import cv2
import numpy as np
import json
from datetime import datetime
from air_quality_detector import AirQualityDetector
import argparse

class RealTimeAirQualityMonitor:
    '''
    Real-time air quality monitoring system
    '''
    
    def __init__(self, model_path):
        '''
        Initialize monitor with trained model
        '''
        self.detector = AirQualityDetector(model_path)
        self.frame_buffer = []
        self.buffer_size = 5
    
    def process_single_image(self, image_path, output_dir='output'):
        '''
        Process single image and generate results
        '''
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing image: {image_path}")
        
        # Get prediction
        result = self.detector.predict_air_quality(image_path)
        
        # Save results to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(output_dir, f'result_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        print(f"\\nResults saved to: {json_path}")
        
        # Create visualization
        viz_path = os.path.join(output_dir, f'visualization_{timestamp}.png')
        self.detector.visualize_prediction(image_path, viz_path)
        
        # Print results
        print("\\n" + "="*60)
        print("AIR QUALITY ASSESSMENT RESULTS")
        print("="*60)
        print(f"Predicted AQI Category: {result['predicted_aqi_category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\\nHaze Features:")
        for key, value in result['haze_features'].items():
            print(f"  {key}: {value:.4f}")
        print("="*60)
        
        return result
    
    def process_image_sequence(self, image_paths, output_dir='output'):
        '''
        Process sequence of 5 consecutive images
        '''
        os.makedirs(output_dir, exist_ok=True)
        
        if len(image_paths) != 5:
            print("Error: Exactly 5 images required for sequence processing")
            return None
        
        print(f"Processing image sequence: {len(image_paths)} images")
        
        # Get prediction
        result = self.detector.predict_sequence(image_paths)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(output_dir, f'sequence_result_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        print(f"\\nResults saved to: {json_path}")
        
        # Print results
        print("\\n" + "="*60)
        print("SEQUENCE-BASED AIR QUALITY ASSESSMENT")
        print("="*60)
        print(f"Predicted AQI Category: {result['predicted_aqi_category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Number of images in sequence: {result['num_images_in_sequence']}")
        print("="*60)
        
        return result
    
    def process_video(self, video_path, output_dir='output', sample_rate=30):
        '''
        Process video file and predict air quality at intervals
        Args:
            video_path: Path to video file
            output_dir: Directory to save results
            sample_rate: Process every Nth frame
        '''
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\\nProcessing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        
        frame_count = 0
        results_list = []
        temp_images = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames
            if frame_count % sample_rate == 0:
                # Save frame temporarily
                temp_path = os.path.join(output_dir, f'temp_frame_{frame_count}.jpg')
                cv2.imwrite(temp_path, frame)
                temp_images.append(temp_path)
                
                # Extract features
                features = self.detector.extract_haze_features(frame)
                
                # Add to buffer for sequence prediction
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
                
                # Make prediction when buffer is full
                if len(self.frame_buffer) == self.buffer_size:
                    # Save frames for prediction
                    sequence_paths = []
                    for i, f in enumerate(self.frame_buffer):
                        seq_path = os.path.join(output_dir, f'seq_frame_{i}.jpg')
                        cv2.imwrite(seq_path, f)
                        sequence_paths.append(seq_path)
                    
                    # Predict
                    result = self.detector.predict_sequence(sequence_paths)
                    result['frame_number'] = frame_count
                    result['timestamp_seconds'] = frame_count / fps
                    result['haze_features'] = features
                    
                    results_list.append(result)
                    
                    print(f"Frame {frame_count}/{total_frames} - "
                          f"AQI: {result['predicted_aqi_category']} "
                          f"({result['confidence']:.1%})")
                    
                    # Clean up sequence frames
                    for path in sequence_paths:
                        if os.path.exists(path):
                            os.remove(path)
        
        cap.release()
        
        # Clean up temp images
        for path in temp_images:
            if os.path.exists(path):
                os.remove(path)
        
        # Save all results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(output_dir, f'video_results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results_list, f, indent=4)
        
        print(f"\\nVideo processing complete. Results saved to: {json_path}")
        print(f"Total predictions: {len(results_list)}")
        
        return results_list
    
    def process_webcam(self, output_dir='output', duration_seconds=60):
        '''
        Real-time processing from webcam
        Args:
            output_dir: Directory to save results
            duration_seconds: Recording duration
        '''
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("\\nStarting webcam monitoring...")
        print(f"Duration: {duration_seconds} seconds")
        print("Press 'q' to quit early")
        
        start_time = datetime.now()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check duration
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > duration_seconds:
                break
            
            # Extract features
            features = self.detector.extract_haze_features(frame)
            
            # Display features on frame
            text_y = 30
            cv2.putText(frame, f"Haze Density: {features['haze_density']:.3f}",
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += 30
            cv2.putText(frame, f"Visibility Score: {features['visibility_score']:.2f} km",
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += 30
            cv2.putText(frame, f"Contrast: {features['contrast']:.2f}",
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add to buffer
            self.frame_buffer.append(frame.copy())
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            
            # Predict every 5 seconds when buffer is full
            if len(self.frame_buffer) == self.buffer_size and frame_count % 150 == 0:
                # Save frames
                sequence_paths = []
                for i, f in enumerate(self.frame_buffer):
                    seq_path = os.path.join(output_dir, f'webcam_frame_{i}.jpg')
                    cv2.imwrite(seq_path, f)
                    sequence_paths.append(seq_path)
                
                # Predict
                result = self.detector.predict_sequence(sequence_paths)
                
                print(f"\\nTime: {elapsed:.1f}s - "
                      f"AQI: {result['predicted_aqi_category']} "
                      f"({result['confidence']:.1%})")
                
                # Clean up
                for path in sequence_paths:
                    if os.path.exists(path):
                        os.remove(path)
            
            # Display frame
            cv2.imshow('Air Quality Monitor', frame)
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\\nWebcam monitoring stopped")

def main():
    parser = argparse.ArgumentParser(
        description='Real-Time Air Quality Assessment System'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['image', 'sequence', 'video', 'webcam'],
                       help='Processing mode')
    parser.add_argument('--input', type=str,
                       help='Input image/video path or comma-separated image paths for sequence')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--sample_rate', type=int, default=30,
                       help='Sample rate for video processing (process every Nth frame)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds for webcam mode')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = RealTimeAirQualityMonitor(args.model)
    
    # Process based on mode
    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        monitor.process_single_image(args.input, args.output)
    
    elif args.mode == 'sequence':
        if not args.input:
            print("Error: --input required for sequence mode")
            return
        image_paths = [p.strip() for p in args.input.split(',')]
        monitor.process_image_sequence(image_paths, args.output)
    
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        monitor.process_video(args.input, args.output, args.sample_rate)
    
    elif args.mode == 'webcam':
        monitor.process_webcam(args.output, args.duration)

if __name__ == "__main__":
    # Example usage without command line args
    print("="*60)
    print("Real-Time Air Quality Assessment System")
    print("="*60)
    print("\\nUsage examples:")
    print("\\n1. Single image:")
    print("   python inference.py --model models/best_model.h5 --mode image --input test_image.jpg")
    print("\\n2. Image sequence:")
    print("   python inference.py --model models/best_model.h5 --mode sequence --input img1.jpg,img2.jpg,img3.jpg,img4.jpg,img5.jpg")
    print("\\n3. Video file:")
    print("   python inference.py --model models/best_model.h5 --mode video --input surveillance_video.mp4")
    print("\\n4. Webcam:")
    print("   python inference.py --model models/best_model.h5 --mode webcam --duration 120")
    print("="*60)
    
    # Uncomment to run with command line arguments
    # main()
"""

with open('inference.py', 'w') as f:
    f.write(inference_script)

print("Inference script saved: inference.py")
