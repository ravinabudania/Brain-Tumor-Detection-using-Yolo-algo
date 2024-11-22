# Brain Tumor Detection Using YOLOv8

## Overview
This project implements a deep learning-based approach to detect brain tumors from medical images. Using the YOLOv8 model, it provides a fast and accurate solution for identifying brain tumor regions, offering potential assistance in medical diagnostics.

## Main Idea
Brain tumor detection is a critical task in medical imaging, and early detection can significantly improve patient outcomes. This project leverages the YOLOv8 model, known for its real-time object detection capabilities, to identify tumor regions in brain scans efficiently.

## Summary
- The model was built using the YOLOv8 framework.
- It processes medical images to detect the presence and location of brain tumors.
- The project includes data preprocessing, model training, and evaluation steps.
- Results are presented visually with annotated images showing detected tumor regions.

## Results
The model achieves high accuracy and provides clear visualizations of tumor detections on test data. Sample outputs include annotated medical scans with bounding boxes around detected tumors.

## Coronal view
![image](https://github.com/user-attachments/assets/5f8d35a1-efd1-4acc-ad8f-bbd2f35064be)


## Sagittal view
![image](https://github.com/user-attachments/assets/07a2f1d8-c61e-4bc9-a1d3-d62c6acfb7fc)


## Axial view
![image](https://github.com/user-attachments/assets/385d46ed-d0e5-4ca2-b3d9-4a78c48b10f2)


## Final Result
![image](https://github.com/user-attachments/assets/54d5b57e-c02f-4ea7-9fda-f7fec4c5d39e)





## Algorithms Used
### YOLOv8 (You Only Look Once, Version 8)
- A state-of-the-art object detection model optimized for speed and accuracy.
- Processes images in a single pass to identify objects (tumors) and their locations.
- Utilizes convolutional neural networks (CNNs) and advanced feature extraction techniques.

### Explanation of the Algorithm
#### YOLO Architecture:
1. Divides the input image into grids.
2. Each grid predicts bounding boxes and confidence scores for objects.
3. Non-max suppression is used to filter overlapping detections.

#### YOLOv8 Specifics:
- Improves upon earlier versions with better performance and reduced latency.
- Supports transfer learning by loading pre-trained weights for initialization.

## Steps Involved
### Data Preparation:
- Images are collected, preprocessed, and split into training, validation, and test sets.

### Model Training:
- YOLOv8 is fine-tuned on the dataset to detect brain tumors.

### Evaluation:
- The model is evaluated for metrics such as precision, recall, and F1 score.

### Visualization:
- Predictions are visualized with bounding boxes on the medical scans.

## How to Run the Project
1. Clone this repository:
    ```bash
    git clone https://github.com/your-repo/brain-tumor-detection-yolov8.git
    cd brain-tumor-detection-yolov8
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the YOLOv8 model or load pre-trained weights:
    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8 model
    ```

4. Run the detection script on test images:
    ```bash
    python detect.py --source path_to_images
    ```

## Future Work
- Extend the model to classify tumor types.
- Enhance the dataset with more diverse and high-resolution images.
- Integrate the solution into a user-friendly application for deployment in hospitals.
