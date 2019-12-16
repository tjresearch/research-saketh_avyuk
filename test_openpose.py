#this file is to test openpose on a few frames, pre-process the data and get it ready to be input data for the actual model
#basically just to refine procedure
#once the full dataset is generated, the procedure can be run on every video there
import pickle
import cv2
import requests
import pandas as pd
import numpy as np
import json
import os
import time
import msgpack
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Pose_Process
from Pose_Process import process
#import sys

#args = sys.argv

#test_frames = args[1]
#print("hello")
from config_reader import config_reader

#params, model_params = config_reader()
test_frames = np.load("frames_1-10.npy")
#print(test_frames.shape)
#print(test_frames[0]['image'].shape)
#print(test_frames[0]['image'])
#plt.imshow(test_frames[0]['image'])
#plt.show()
raw_data = [(x['image'],x['isFight']) for x in test_frames]
plt.imshow(process(raw_data[0][0]))
plt.show()
#plt.imshow(test_frames[0]['image'])
#plt.show()
print(np.array(raw_data).shape)