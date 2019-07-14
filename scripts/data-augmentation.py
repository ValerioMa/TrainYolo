#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import os
import cv2
import struct
import imghdr
from data_aug.data_aug import *
from data_aug.bbox_util import *


# In[2]:


base_url = "{}/../data/images/".format(os.path.dirname(os.path.realpath(__file__)))


# In[3]:


cls_index_to_name = {
    0: "VOID",
    1: "Yellow Cone",
    2: "Blue Cone",
    3: "Orange Cone"
}


# In[4]:


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height


# In[5]:


def store_parsed_label_data(train_image_file_name, label_file_name, image_labels):
    parsed_train_data = {
        "file_name": train_image_file_name,
        "labels": image_labels
    }
    
    with open(label_file_name, "w") as label_file:
        json.dump(parsed_train_data, label_file)


# In[6]:


def convert_augmented_bboxes_to_original_format(train_file_name, augmented_bboxes):
    bboxes = []
    
    for augmented_bbox in augmented_bboxes:
        x1 = augmented_bbox[0]
        y1 = augmented_bbox[1]
        x2 = augmented_bbox[2]
        y2 = augmented_bbox[3]
        cls_index = augmented_bbox[4]
        cls_name = cls_index_to_name[cls_index] 
        
        unnormalized_width = x2 - x1
        unnormalized_height = y2 - y1
        img_width, img_height = get_image_size(train_file_name)
        width = unnormalized_width / img_width
        height = unnormalized_height / img_height
        x1_normalized = x1 / img_width
        y1_normalized = y1 / img_height 
        
        label_data = {
            "x": x1_normalized,
            "y": y1_normalized,
            "x_unnormalized": x1,
            "y_unnormalized": y1,
            "width": width,
            "height": height,
            "width_unnormalized": unnormalized_width,
            "height_unnormalized": unnormalized_height,
            "class_index": cls_index,
            "class_name": cls_index_to_name[cls_index]
        }
        
        bboxes.append(label_data)
    
    return bboxes


# In[7]:


def convert_bboxes_to_augmentation_format(original_bboxes):
    bboxes = []
    
    for original_bbox in original_bboxes:
        x1 = original_bbox["x_unnormalized"]
        y1 = original_bbox["y_unnormalized"]
        x2 = x1 + original_bbox["width_unnormalized"]
        y2 = y1 + original_bbox["height_unnormalized"]
        cls_index = original_bbox["class_index"]
        
        bboxes.append([x1, y1, x2, y2, cls_index])
    
    bboxes = np.array(bboxes, dtype=np.float64)
    
    return bboxes


# In[8]:


def get_label_files(dir_path):
    files = []
    for r, d, f in os.walk(dir_path):
        for file in f:
            if '.labels' in file:
                files.append(os.path.join(r, file))
    
    return files


# In[9]:


for subdir, dirs, files in os.walk(base_url):
    if base_url != subdir:
        label_files_names =  get_label_files(subdir)
        for label_file_name in label_files_names:
            train_file_name = label_file_name.split(".labels")[0]
            label_file_name = "{}.parsed_labels".format(train_file_name)
            
            original_label_file = open(label_file_name, "r")
            original_label_file_data = original_label_file.read()
            original_label_file.close()
            original_label_json_data = json.loads(original_label_file_data)
            
            img = cv2.imread(train_file_name)[:,:,::-1]
            bboxes = convert_bboxes_to_augmentation_format(original_label_json_data["labels"])
            
            # flip image horizontally
            img_flipped, bboxes_flipped = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
            train_file_name_flipped = "{}_flipped.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_flipped = "{}_flipped.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_flipped = cv2.cvtColor(img_flipped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_flipped, img_flipped)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_flipped, bboxes_flipped)
            store_parsed_label_data(train_file_name_flipped, label_file_name_flipped, image_labels)
            
            # scale image
            img_sclaed, bboxes_scaled = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())
            train_file_name_scaled = "{}_scaled.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_scaled = "{}_scaled.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_sclaed = cv2.cvtColor(img_sclaed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_scaled, img_sclaed)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_scaled, bboxes_scaled)
            store_parsed_label_data(train_file_name_scaled, label_file_name_scaled, image_labels)
            
            # translate image
            img_translated, bboxes_translated = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
            train_file_name_translated = "{}_translated.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_translated = "{}_translated.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_translated = cv2.cvtColor(img_translated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_translated, img_translated)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_translated, bboxes_translated)
            store_parsed_label_data(train_file_name_translated, label_file_name_translated, image_labels)
            
            # rotate image
            img_rotated, bboxes_rotated = RandomRotate(20)(img.copy(), bboxes.copy())
            train_file_name_rotated = "{}_rotated.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_rotated = "{}_rotated.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_rotated, img_rotated)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_rotated, bboxes_rotated)
            store_parsed_label_data(train_file_name_rotated, label_file_name_rotated, image_labels)
            
            # shear image
            img_sheared, bboxes_sheared = RandomShear(0.2)(img.copy(), bboxes.copy())
            train_file_name_sheared = "{}_sheared.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_sheared = "{}_sheared.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_sheared = cv2.cvtColor(img_sheared, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_sheared, img_sheared)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_sheared, bboxes_sheared)
            store_parsed_label_data(train_file_name_sheared, label_file_name_sheared, image_labels)
            
            #resize image to square dimensions with half the height
            img_resized_small, bboxes_resized_small = Resize(512)(img.copy(), bboxes.copy())
            train_file_name_resized_small = "{}_resized_small.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_resized_small = "{}_resized_small.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_resized_small = cv2.cvtColor(img_resized_small, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_resized_small, img_resized_small)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_resized_small, bboxes_resized_small)
            store_parsed_label_data(train_file_name_resized_small, label_file_name_resized_small, image_labels)
            
            #resize image to square dimensions with double the height
            img_resized_large, bboxes_resized_large = Resize(2048)(img.copy(), bboxes.copy())
            train_file_name_resized_large = "{}_resized_large.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_resized_large = "{}_resized_large.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_resized_large = cv2.cvtColor(img_resized_large, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_resized_large, img_resized_large)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_resized_large, bboxes_resized_large)
            store_parsed_label_data(train_file_name_resized_large, label_file_name_resized_large, image_labels)
            
            # transform HSV color values
            img_hsv_transform, bboxes_hsv_transform = RandomHSV(100, 100, 100)(img.copy(), bboxes.copy())
            train_file_name_hsv_transform = "{}_hsv_transform.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_hsv_transform = "{}_hsv_transform.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_hsv_transform = cv2.cvtColor(img_hsv_transform, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_hsv_transform, img_hsv_transform)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_hsv_transform, bboxes_hsv_transform)
            store_parsed_label_data(train_file_name_hsv_transform, label_file_name_hsv_transform, image_labels)
            
            # combine multiple augmentation methods
            seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomRotate(10), RandomScale(), RandomTranslate(), RandomShear()])
            img_combined_changes, bboxes_combined_changes = seq(img.copy(), bboxes.copy())
            train_file_name_combined_changes = "{}_combined_changes.jpg".format(train_file_name.split(".jpg")[0])
            label_file_name_combined_changes = "{}_combined_changes.jpg.parsed_labels".format(train_file_name.split(".jpg")[0])
            img_combined_changes = cv2.cvtColor(img_combined_changes, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_file_name_combined_changes, img_combined_changes)
            image_labels = convert_augmented_bboxes_to_original_format(train_file_name_combined_changes, bboxes_combined_changes)            
            store_parsed_label_data(train_file_name_combined_changes, label_file_name_combined_changes, image_labels)

