#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import struct
import imghdr
import json
import os


# In[2]:


base_url = "{}/images/".format(os.getcwd())


# In[3]:


def extract_training_labels(label_data, train_file_path):
    min_x = 999999999999
    max_x = -1
    min_y = 999999999999
    max_y = -1
    
    for vertex in label_data["polygon"]:
        if vertex["x"] < min_x:
            min_x = vertex["x"]
        if vertex["y"] < min_y:
            min_y = vertex["y"]
        if vertex["x"] > max_x:
            max_x = vertex["x"]
        if vertex["y"] > max_y:
            max_y = vertex["y"]
    
    train_image_width, train_image_height = get_image_size(train_file_path)
    
    min_x_normalized = min_x / train_image_width
    min_y_normalized = min_y / train_image_height
    
    bounding_box_width = (max_x - min_x) / train_image_width
    bounding_box_height = (max_y - min_y) / train_image_height
    
    bounding_box_width_unnnormalized = (max_x - min_x)
    bounding_box_height_unnnormalized = (max_y - min_y)
    
    cls_index = label_data["classIndex"]
    cls_name = label_data["label"]
    
    return min_x_normalized, min_y_normalized, min_x, min_y, bounding_box_width, bounding_box_height, bounding_box_width_unnnormalized, bounding_box_height_unnnormalized, cls_index, cls_name


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


def get_label_files(dir_path):
    files = []
    for r, d, f in os.walk(dir_path):
        for file in f:
            if '.labels' in file:
                files.append(os.path.join(r, file))
    
    return files


# In[6]:


def store_parsed_label_data(train_image_file_name, image_labels):
    parsed_train_data = {
        "file_name": train_image_file_name,
        "labels": []
    }
    
    for label in image_labels:
        (min_x_normalized, min_y_normalized, min_x, min_y, bounding_box_width, bounding_box_height, bounding_box_width_unnnormalized, bounding_box_height_unnnormalized, cls_index, cls_name) = label
        label_data = {
            "x": min_x_normalized,
            "y": min_y_normalized,
            "x_unnormalized": min_x,
            "y_unnormalized": min_y,
            "width": bounding_box_width,
            "height": bounding_box_height,
            "width_unnormalized": bounding_box_width_unnnormalized,
            "height_unnormalized": bounding_box_height_unnnormalized,
            "class_index": cls_index,
            "class_name": cls_name
        }
        parsed_train_data["labels"].append(label_data)
    
    parsed_label_file_name = "{}.parsed_labels".format(train_image_file_name)
    with open(parsed_label_file_name, "w") as parsed_label_file:
        json.dump(parsed_train_data, parsed_label_file)


# In[7]:


for subdir, dirs, files in os.walk(base_url):
    label_files_names =  get_label_files(subdir)
    for label_file_name in label_files_names:
        train_image_file_name = label_file_name.split(".labels")[0]
        original_label_file = open(label_file_name, "r")
        original_label_file_data = original_label_file.read()
        original_label_file.close()
        original_label_json_data = json.loads(original_label_file_data)

        image_labels = []
        for i in range(len(original_label_json_data["objects"])):
            min_x_normalized, min_y_normalized, min_x, min_y, bounding_box_width, bounding_box_height, bounding_box_width_unnnormalized, bounding_box_height_unnnormalized, cls_index, cls_name = extract_training_labels(original_label_json_data["objects"][i], train_image_file_name)
            image_labels.append((min_x_normalized, min_y_normalized, min_x, min_y, bounding_box_width, bounding_box_height, bounding_box_width_unnnormalized, bounding_box_height_unnnormalized, cls_index, cls_name))

        store_parsed_label_data(train_image_file_name, image_labels)

