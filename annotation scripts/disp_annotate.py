import numpy as np
import json
import cv2

# Function to load bounding box coordinates from .npy file
def load_bbox_from_npy(file_path):
    return np.load(file_path)

# Function to load label from .json file
def load_label_from_json(file_path):
    with open(file_path, 'r') as f:
        label_data = json.load(f)
    return label_data['label']

# Function to display image with bounding boxes and labels
def display_image_with_annotations(image_path, bbox_coordinates, label):
    image = cv2.imread(image_path)
    for bbox in bbox_coordinates:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Annotated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Iterate through each file and annotate
for i in range(1, 101):  # Assuming you have 100 files
    image_path = f'rgb{str(i).zfill(3)}.png'
    bbox_file_path = f'bounding_box_{str(i).zfill(3)}.npy'
    label_file_path = f'label_{str(i).zfill(3)}.json'
    
    bbox_coordinates = load_bbox_from_npy(bbox_file_path)
    label = load_label_from_json(label_file_path)
    
    display_image_with_annotations(image_path, bbox_coordinates, label)
