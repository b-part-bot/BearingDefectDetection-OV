import cv2
import numpy as np
import json
import os
import zipfile

def load_labels(file_path):
    with open(file_path, 'r') as f:
        label_data = json.load(f)
    return label_data

def convert_bbox_to_yolo(bbox, img_width, img_height):
    _, x1, y1, x2, y2, _ = bbox
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return (x_center, y_center, width, height)

def display_image_with_annotations(image_path, bbox_coordinates, labels):
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    for bbox, label in zip(bbox_coordinates, labels):
        x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow(image)

def process_files(image_dir, bbox_dir, label_dir, output_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            img_height, img_width = image.shape[:2]
            base_name = os.path.splitext(filename)[0]

            bbox_file_path = os.path.join(bbox_dir, f'bounding_box_2d_tight_{base_name[4:]}.npy')
            label_file_path = os.path.join(label_dir, f'bounding_box_2d_tight_labels_{base_name[4:]}.json')

            bboxes = np.load(bbox_file_path)
            labels = load_labels(label_file_path)

            yolo_data = []
            for bbox in bboxes:
                class_id = int(bbox[0])  # Assuming class ID is the first number in bbox
                yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                yolo_data.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")

            output_path = os.path.join(output_dir, f'{base_name}.txt')
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_data))

def zip_yolo_with_images(yolo_dir, image_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for filename in os.listdir(yolo_dir):
            if filename.endswith('.txt'):
                yolo_file_path = os.path.join(yolo_dir, filename)
                image_file_path = os.path.join(image_dir, os.path.splitext(filename)[0] + '.png')
                zipf.write(yolo_file_path, arcname=os.path.join('annotations', filename))
                zipf.write(image_file_path, arcname=os.path.join('images', os.path.basename(image_file_path)))

def clear_directory(dir_path):
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)

# Define your directories and output zip file
image_dir = 'current/outer_4'
bbox_dir = 'current/outer_4'
label_dir = 'current/outer_4'
output_dir = 'current/output_txt'
output_zip = 'current/ouput_zips/outer_4_zip_file.zip'

process_files(image_dir, bbox_dir, label_dir, output_dir)
zip_yolo_with_images(output_dir, image_dir, output_zip)
clear_directory(output_dir)
