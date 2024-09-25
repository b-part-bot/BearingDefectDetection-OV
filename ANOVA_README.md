
# Bearing Defect Detection Dataset and Model Training

This repository contains details for creating datasets, training models, and performing ANOVA analysis on results. The objective is to detect defects in bearings using various lighting and camera angle combinations. This guide walks through the process of dataset creation, model training, and analysis of results using ANOVA.

## Dataset

### Access Links

- **Baseline Dataset:** [Roboflow Link](https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6Ijd2bGhqWUdjUkZlS2JQaG02TGgxMnlXTHNTVjIiLCJyb2xlIjoib3duZXIiLCJpbnZpdGVyIjoiYnBhcnRoYTEyNTRAZ21haWwuY29tIiwiaWF0IjoxNzI3Mjk4NDg3fQ.u_D1dv-h9M-ZrUxI5OwCVeAc7S5e3W7ZWUFdnqmZx9s)
- **ANOVA Dataset:** [Roboflow Link](https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6Im11QndOT3RtZ0pnOHhtNk1jYXBuIiwicm9sZSI6Im93bmVyIiwiaW52aXRlciI6ImJwYXJ0aGExMjU0QGdtYWlsLmNvbSIsImlhdCI6MTcyNzI5ODMwMX0.H-7DhLibmuSCxqFA_G0xc1RoyBT65sjaEzgQhzbulv4)

### Dataset Details

- **Baseline Dataset:** [Link](https://app.roboflow.com/faultdetection-uasup/faultdetection-wkv2r/1)
- **ANOVA Dataset:** [Link](https://app.roboflow.com/bearingfaultdetection)
- Nine datasets for each combination of `[L1, L2, L3] x [A1, A2, A3]`
- One common validation dataset

## Dataset Creation Process

1. Install **Nvidia Omniverse Beta** version.
2. Install **Omniverse Replicator**.
3. Use the 3D model of a bearing from [this repository](https://github.com/b-part-bot/BearingDefectDetection-OV/tree/main/current_models).
4. Adjust parameters so that the scratch is visible on the bearing.
5. Generate images under different lighting and camera angles.
6. Images will be saved in the `/output` folder as `.png` files, with corresponding `.json` files containing bounding box coordinates for the defect.
7. Run the [export script](https://github.com/b-part-bot/BearingDefectDetection-OV/tree/main/annotation%20scripts) to convert images and bounding boxes to YOLOv9 format.
8. Create a new project in Roboflow and import the folders one by one (this is important since file names will be the same across folders).
9. Verify that all images are properly annotated.
10. Generate the complete dataset with any necessary augmentations or preprocessing steps.

## Model Training

### Baseline Model Training

1. Import the YOLOv9 dataset and ensure that the train/validation/test partitions are correctly set up.
2. Train the baseline model using YOLOv9. Instructions can be found in the [Model Training guide](https://github.com/b-part-bot/BearingDefectDetection-OV/tree/main/Model%20Training).

### ANOVA Model Training

To perform ANOVA analysis, we need to train at least two models on the same dataset and compare their performance on a common validation set.

#### Steps:

1. For each combination of lighting and camera angles (L1A1, L1A2, ..., L3A3), download the dataset and the common validation set.
2. Train two models with different hyperparameters.
3. Evaluate both models using the common validation set.
4. Record the results for each combination.

## ANOVA Analysis

Perform ANOVA analysis on the recorded results to determine:

- The best combination of lighting and camera angles.
- The impact of lighting versus camera angle on model performance.
