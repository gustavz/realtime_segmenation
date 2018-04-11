#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:17:38 2018

@author: github.com/GustavZ
"""
import os
import tarfile
from six.moves import urllib
import numpy as np
import tensorflow as tf
import yaml
import cv2
from stuff.helper import TimeLiner
from tensorflow.python.client import timeline

## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        
VIDEO_INPUT		= cfg['video_input']
FPS_INTERVAL		= cfg['fps_interval']
ALPHA			= cfg['alpha']
MODEL_NAME		= cfg['model_name']
MODEL_PATH		= cfg['model_path']
DOWNLOAD_BASE		= cfg['download_base']
IMAGE_PATH		= cfg['image_path']
CPU_ONLY 		= cfg['cpu_only']

if CPU_ONLY:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	DEVICE = '_CPU'
else:
	DEVICE = '_GPU'


# Hardcoded COCO_VOC Labels
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'])

def create_colormap(seg_map):
	"""
	Takes A 2D array storing the segmentation labels.
	Returns A 2D array where each element is the color indexed 
	by the corresponding element in the input label to the PASCAL color map.
	"""
	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)
	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((ind >> channel) & 1) << shift 
		ind >>= 3
	return colormap[seg_map]

# Download Model from TF-deeplab's Model Zoo
def download_model():
	model_file = MODEL_NAME + '.tar.gz'
	if not os.path.isfile(MODEL_PATH):
		print('> Model not found. Downloading it now.')
		opener = urllib.request.URLopener()
		opener.retrieve(DOWNLOAD_BASE + model_file, model_file)
		tar_file = tarfile.open(model_file)
		for file in tar_file.getmembers():
		  file_name = os.path.basename(file.name)
		  if 'frozen_inference_graph.pb' in file_name:
		    tar_file.extract(file, os.getcwd() + '/models/')
		os.remove(os.getcwd() + '/' + model_file)
	else:
		print('> Model found. Proceed.')
        
# Visualize Text on OpenCV Image
def vis_text(image,string,pos):
	cv2.putText(image,string, (10,30*pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

# Load frozen Model   
def load_frozenmodel():
	print('> Loading frozen model into memory')
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  seg_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
		serialized_graph = fid.read()
		seg_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(seg_graph_def, name='')
	return detection_graph


def segmentation(detection_graph,label_names):
	# load images
	images = []
	for root, dirs, files in os.walk(IMAGE_PATH):
		for file in files:
		    if file.endswith(".jpg"):
		        images.append(os.path.join(root, file))
	images.sort()
	# TF config
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	print("> Starting Segmentaion")
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
		    for im in images:
		    	# input
		        image = cv2.imread(im)
		        height, width, channels = image.shape
		        resize_ratio = 1.0 * 513 / max(width,height)
		        target_size = (int(resize_ratio * width), int(resize_ratio * height))
		        resized_image = cv2.resize(image, target_size)
		        # TF session + timing
		        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		        run_metadata = tf.RunMetadata()
		        many_runs_timeline = TimeLiner()
		        batch_seg_map = sess.run('SemanticPredictions:0',
		        				feed_dict={'ImageTensor:0': [cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)]},
		        				options=options, run_metadata=run_metadata)
		        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
		        chrome_trace = fetched_timeline.generate_chrome_trace_format()
		        many_runs_timeline.update_timeline(chrome_trace)
		        with open('test_timeline{}.json'.format(DEVICE), 'w') as f:
		        	f.write(chrome_trace)
		        # visualization
		        seg_map = batch_seg_map[0]
		        seg_image = create_colormap(seg_map).astype(np.uint8)
		        labels = label_names[np.unique(seg_map)]
		        cv2.addWeighted(seg_image,ALPHA,resized_image,1-ALPHA,0,resized_image)
		        vis_text(image,"labels: {}".format(labels),1)
		        cv2.imshow('segmentation',resized_image)
		        if cv2.waitKey(2000) & 0xFF == ord('q'):
		        	break
		        # DEBUG
		        print ("\n{}".format(im))
		        print ("seg_map:\n{}".format(seg_map))
		        print ("unique seg_map:\n{}".format(np.unique(seg_map)))
	cv2.destroyAllWindows()

def main():
	download_model()
	graph = load_frozenmodel()
	segmentation(graph, LABEL_NAMES)
      
if __name__ == '__main__':
    main()  
