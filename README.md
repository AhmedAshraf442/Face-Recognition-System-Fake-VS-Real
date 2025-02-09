# Face-Recognition-System-Fake-VS-Real

A Computer Vision project designed to distinguish between live and pre-recorded face images using a custom dataset and a YOLOv8n-based model.

## Overview

The **Face-Recognition-System-Fake-VS-Real** project tackles the challenge of differentiating between live images and pre-recorded ones. The project pipeline includes dataset collection, face detection, image preprocessing, model training, and evaluation. Key techniques involve enhancing the bounding box to capture the entire head, filtering out blurry images, normalizing coordinates for YOLO, and shuffling data to ensure randomness.

## Features

- **Face Detection:** Uses a Face Detector to track faces and capture images with their coordinates.
- **Bounding Box Enhancement:** Enlarges the bounding box by a specific percentage on all sides to include the whole head.
- **Image Quality Control:** Filters out images that don't meet the clarity threshold based on blurriness measurement.
- **YOLO Formatting:** Normalizes coordinates and calculates `xc` and `yc` to prepare data for YOLO detection.
- **Dataset Collection:** Captures images via a webcam and assigns labels (`fake` or `real`) accordingly.
- **Data Splitting:** Divides the dataset into 70% training, 20% validation, and 10% testing.
- **Model Training:** Fine-tunes a pretrained YOLOv8n model over 20 epochs, achieving 98% mAP50 and a 0.2 class loss.
- **Data Shuffling:** Randomizes data to avoid consecutive frames that may be too similar.

## Demo Video

Watch the demonstration video below for a quick walkthrough of the project:

[Watch the Video](https://drive.google.com/file/d/1DzK-9MhFafk4srEm-7WIandsxiK2zMTJ/view?usp=sharing)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Face-Recognition-System-Fake-VS-Real.git
   cd Face-Recognition-System-Fake-VS-Real
