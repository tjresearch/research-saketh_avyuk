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


test_frames = np.load("frames_1-10.npy")
#print(test_frames.shape)
#print(test_frames[0]['image'].shape)
#print(test_frames[0]['image'])

plt.imshow(test_frames[0]['image'])
plt.show()