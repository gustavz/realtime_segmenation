#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:17:38 2018

@author: gustav
"""
import os
import tarfile
from six.moves import urllib
import numpy as np
import tensorflow as tf
import yaml
import cv2
from stuff.helper import FPS2, WebcamVideoStream


## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        
VIDEO_INPUT         = cfg['video_input']
FPS_INTERVAL       	= cfg['fps_interval']
ALPHA				= cfg['alpha']
MODEL_NAME          = cfg['model_name']
MODEL_PATH          = cfg['model_path']
DOWNLOAD_BASE		= cfg['download_base']

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
	# fixed input sizes as model needs resize either way
	vs = WebcamVideoStream(VIDEO_INPUT,640,480).start() 
	resize_ratio = 1.0 * 513 / max(vs.real_width,vs.real_height)
	target_size = (int(resize_ratio * vs.real_width), int(resize_ratio * vs.real_height))
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	fps = FPS2(FPS_INTERVAL).start()
	print("> Starting Segmentaion")
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
		    while vs.isActive():
		        image = cv2.resize(vs.read(),target_size)
		        batch_seg_map = sess.run('SemanticPredictions:0',
		        				feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})
		        seg_map = batch_seg_map[0]
		        seg_image = create_colormap(seg_map).astype(np.uint8)
		        labels = label_names[np.unique(seg_map)]
		        # visualization
		        cv2.addWeighted(seg_image,ALPHA,image,1-ALPHA,0,image)
		        vis_text(image,"fps: {}".format(fps.fps_local()),1)
		        vis_text(image,"labels: {}".format(labels),2)
		        cv2.imshow('segmentation',image)
		        if cv2.waitKey(1) & 0xFF == ord('q'):
		        	break	
		        fps.update()
	fps.stop()
	vs.stop()
	cv2.destroyAllWindows()

def main():
	download_model()
	graph = load_frozenmodel()
	segmentation(graph, LABEL_NAMES)
      
if __name__ == '__main__':
    main()  
