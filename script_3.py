
# Create comprehensive dataset information document

dataset_info = """# Air Quality Assessment Dataset Information

## Overview
This document provides detailed information about the datasets used for training the Real-Time Urban Air Quality Assessment model.

## Primary Datasets

### 1. HVAQ Dataset (High-Resolution Vision-Based Air Quality)
**Source:** IEEE Transactions on Instrumentation and Measurement, 2021
**Access:** https://github.com/implicitDeclaration/HVAQ-dataset
**Google Drive:** Available through GitHub repository

**Description:**
- High temporal and spatial resolution air quality dataset
- Simultaneous point sensor measurements and corresponding images
- Contains PM2.5, PM10, temperature, and humidity data
- Collection period: 10:00-15:00 on 3 different days

**Features:**
- Image resolution: High-resolution surveillance camera images
- Air quality metrics: PM2.5, PM10 concentrations
- Meteorological data: Temperature, humidity
- Temporal resolution: Minute-level measurements

**Citation:**
```
@ARTICLE{9546708,
  author={Chen, Zuohui and Zhang, Tony and Chen, Zhuangzhi and Xiang, Yun and Xuan, Qi and Dick, Robert P.},
  journal={IEEE Transactions on Instrumentation and Measurement},
  title={HVAQ: A High-Resolution Vision-Based Air Quality Dataset},
  year={2021},
  volume={70},
  pages={1-10},
  doi={10.1109/TIM.2021.3104415}
}
```

---

### 2. TRAQID Dataset (Traffic-Related Air Quality Image Dataset)
**Source:** CVIT, IIIT Hyderabad
**Access:** https://github.com/TRAQID/TRAQID
**Paper:** Available on ACM Digital Library

**Description:**
- 26,678 traffic images with sensor data
- Collected in Hyderabad and Secunderabad, India
- Both daytime and nighttime captures
- Front and rear traffic imagery

**Features:**
- PM2.5 and PM10 concentrations
- Air Quality Index (AQI)
- Temperature and humidity
- Diverse visual conditions (day/night, weather variations)

**Image Specifications:**
- Resolution: Varied (preprocessed to 224x224 for training)
- Capture locations: Urban traffic zones
- Time span: Multiple months of continuous monitoring

---

### 3. PM25Vision Dataset
**Source:** ArXiv 2509.16519, September 2025
**Access:** Available on Kaggle - https://www.kaggle.com/datasets/pm25vision
**Paper:** https://arxiv.org/abs/2509.16519

**Description:**
- Largest benchmark dataset for PM2.5 estimation from images
- 11,114+ images matched with timestamped PM2.5 readings
- 3,261 AQI monitoring stations
- 11 years of data (spatial accuracy: 5 kilometers)

**Features:**
- Street-level imagery
- Geolocated PM2.5 readings
- Multi-year temporal coverage
- CNN and Transformer baseline models provided

**Citation:**
```
Han, Yang. "PM25Vision: A Large-Scale Benchmark Dataset for Visual Estimation of Air Quality."
arXiv preprint arXiv:2509.16519 (2025).
```

---

### 4. SAPID (Smartphone-Based Air Pollution Image Dataset)
**Source:** Mendeley Data
**Access:** https://data.mendeley.com/datasets/p7yb87hym5/1

**Description:**
- 456 images divided into five AQI categories
- Categories follow United States Environmental Protection Agency standards
- Captured using smartphones
- Suitable for mobile air quality monitoring applications

**AQI Categories:**
1. Good (0-50)
2. Moderate (51-100)
3. Unhealthy for Sensitive Groups (101-150)
4. Unhealthy (151-200)
5. Very Unhealthy (201-300)

---

### 5. Beijing PM2.5 Dataset
**Source:** UCI Machine Learning Repository
**Access:** https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
**Alternative:** Kaggle - https://www.kaggle.com/datasets/sid321axn/beijing-pm25-air-pollution

**Description:**
- Hourly PM2.5 data from US Embassy in Beijing
- Time period: January 1, 2010 to December 31, 2014
- Meteorological data from Beijing Capital International Airport
- 43,824 instances with 11 features

**Features:**
- PM2.5 concentration (μg/m³)
- Meteorological variables: Temperature, Pressure, Dew Point
- Wind speed and direction
- Precipitation (rain, snow)

---

### 6. Air Quality Image Dataset from India and Nepal
**Source:** Kaggle
**Access:** https://www.kaggle.com/datasets/shivamb/air-quality-image-dataset

**Description:**
- Multi-country dataset covering India and Nepal
- Images categorized into 6 AQI levels
- Resolution: 224x224 pixels (preprocessed)
- Diverse environmental conditions

**Categories:**
- Good
- Moderate  
- Satisfactory
- Poor
- Very Poor
- Severe

---

## Secondary/Supporting Datasets

### 7. HazyDet Dataset
**Source:** GitHub (https://github.com/GrokCV/HazyDet)
**Purpose:** Haze detection and object detection in hazy conditions

**Description:**
- 383,000 instances for drone-view object detection
- Real-world hazy captures and synthetic augmentation
- Depth information for atmospheric analysis

---

### 8. KITTI Dataset (for Depth Estimation)
**Source:** http://www.cvlibs.net/datasets/kitti/
**Purpose:** Ground truth for depth map generation

**Description:**
- High-resolution images with LiDAR depth data
- Used for generating synthetic hazy images
- Urban driving scenarios

---

## Dataset Preparation

### Image Preprocessing Pipeline
1. **Resizing:** All images resized to 224×224 pixels
2. **Normalization:** Pixel values scaled to [0, 1]
3. **Color Space:** RGB format
4. **Augmentation:**
   - Random horizontal flip
   - Random rotation (±10°)
   - Random zoom (±10%)
   - Random contrast adjustment

### Sequence Creation
For temporal analysis (CNN-LSTM model):
- **Sequence length:** 5 consecutive frames
- **Temporal gap:** 30 seconds to 1 hour (depending on dataset)
- **Overlap:** 50% between sequences for training augmentation

### Label Encoding
AQI categories mapped to numerical labels:
```python
{
    0: 'Good (0-50)',
    1: 'Moderate (51-100)',
    2: 'Unhealthy for Sensitive Groups (101-150)',
    3: 'Unhealthy (151-200)',
    4: 'Very Unhealthy (201-300)',
    5: 'Hazardous (>300)'
}
```

---

## Data Split

### Training/Validation/Test Split
- **Training:** 70% of data
- **Validation:** 15% of data  
- **Testing:** 15% of data

### Stratification
- Stratified sampling ensures balanced representation of all AQI categories
- Temporal consistency maintained (sequences not split across sets)

---

## Download Instructions

### HVAQ Dataset
```bash
# Clone repository
git clone https://github.com/implicitDeclaration/HVAQ-dataset.git
cd HVAQ-dataset

# Download from Google Drive link provided in repository README
```

### TRAQID Dataset
```bash
# Clone repository
git clone https://github.com/TRAQID/TRAQID.git
cd TRAQID

# Follow instructions in README for dataset download
```

### PM25Vision Dataset
```bash
# Download from Kaggle
kaggle datasets download -d pm25vision/global-images-pm25

# Or visit: https://www.kaggle.com/datasets/pm25vision
```

### SAPID Dataset
```bash
# Download from Mendeley Data
# Visit: https://data.mendeley.com/datasets/p7yb87hym5/1
# Direct download available after registration
```

### Beijing PM2.5 Dataset
```bash
# From UCI ML Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv

# Or from Kaggle
kaggle datasets download -d sid321axn/beijing-pm25-air-pollution
```

---

## Dataset Statistics

### Combined Dataset Summary
| Dataset | Images | Categories | Resolution | Temporal | Sensor Data |
|---------|--------|------------|------------|----------|-------------|
| HVAQ | ~5,000 | Continuous | High | Yes | Yes |
| TRAQID | 26,678 | 6 | Varied | Yes | Yes |
| PM25Vision | 11,114 | Continuous | High | Yes | Yes |
| SAPID | 456 | 5 | 224×224 | No | No |
| Beijing PM2.5 | 43,824 | Continuous | - | Yes | Yes |
| India-Nepal | ~10,000 | 6 | 224×224 | No | No |

### Total Training Data
- **Images:** ~56,000+
- **Sequences (5-frame):** ~11,000+
- **Categories:** 6 AQI levels
- **Augmented samples:** ~33,000+ (3× augmentation)

---

## Citation Guidelines

When using these datasets, please cite the respective papers:

**HVAQ:**
Chen, Z., et al. (2021). "HVAQ: A High-Resolution Vision-Based Air Quality Dataset."
IEEE Transactions on Instrumentation and Measurement, 70, 1-10.

**PM25Vision:**
Han, Y. (2025). "PM25Vision: A Large-Scale Benchmark Dataset for Visual Estimation 
of Air Quality." arXiv preprint arXiv:2509.16519.

**TRAQID:**
Cite the ACM publication when accessing from GitHub repository.

**Beijing PM2.5:**
Liang, X., et al. (2015). "Assessing Beijing's PM2.5 pollution." 
Proceedings of the Royal Society A.

---

## Additional Resources

### Related Datasets
- **4K Dehazing Dataset:** For image enhancement research
- **RESIDE Dataset:** Synthetic and real-world hazy images
- **Dense-HAZE:** Dense haze image dataset

### Tools and Libraries
- **Labelme:** For manual image annotation
- **CVAT:** For video annotation and tracking
- **RoboFlow:** Dataset management and augmentation

---

## Contact and Support

For dataset-specific questions:
- HVAQ: Contact authors via IEEE paper
- TRAQID: GitHub issues (https://github.com/TRAQID/TRAQID/issues)
- PM25Vision: Kaggle dataset discussion forum
- SAPID: Mendeley Data platform

For project-specific questions:
Contact project maintainers through project repository.

---

**Last Updated:** November 2025
**Version:** 1.0
"""

with open('DATASET_INFO.md', 'w') as f:
    f.write(dataset_info)

print("Dataset information saved: DATASET_INFO.md")
print("=" * 60)
