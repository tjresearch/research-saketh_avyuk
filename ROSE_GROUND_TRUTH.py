#TO DO:
# download everything and run this jaunt
#NOTE- don't need to crop or resize because represenation of poses should be relative

#******FIX THIS SO THAT ITS EASY TO DISTINGUISH BETWEEN DIFFERENT VIDEOS****

#switch from pickle to something else - https://shocksolution.com/2010/01/10/storing-large-numpy-arrays-on-disk-python-pickle-vs-hdf5adsf/

import matplotlib
import pickle
import cv2
import requests
import pandas as pd
import numpy as np
import json
import os
import time
import msgpack

response = requests.get("http://rose1.ntu.edu.sg/Datasets/NTU%20CCTV-Fights%20Dataset/download/groundtruth.json.txt")
ground_truth = json.loads(response.content)

#database key has all the shit in it
#every video labelled as fight_0001 through fight_1000
#
#every video has frames which are labelled with or without fights
#this becomes the initial dataset and has to be run basically


#testing with a single video right now

VIDEOS = os.listdir("ROSE_1-800")
ROOT = os.getcwd()
#print(os.path.join(ROOT+"\\"+VIDEOS[0]))

frames = []
vc = 0
for video in VIDEOS:
    start = time.time()
    print("-----    VIDEO:  ",video,"  -----")

    index = video.split(".")[0]
    fight_ground_truth = ground_truth["database"][index]
    fps = fight_ground_truth["frame_rate"]
    num_frames = fight_ground_truth["nb_frames"]
    isCCTV = True if fight_ground_truth["source"] == "CCTV" else False

    fights = []
    for a in fight_ground_truth["annotations"]:
        if a["label"].upper() == "FIGHT":
            fights.append(range(*[int(float(fps) * x) for x in a["segment"]]))

    cap = cv2.VideoCapture(os.path.join(ROOT,"ROSE_1-800",video))
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print("CURRENT FRAME ", len(frames))
        #used to test whether or not the image was actually legit
        # it is

        #from matplotlib import pyplot as plt
        #plt.imshow(frame)
        #plt.show()

        if ret == True:
            isFight = 0
            for f in fights:
                if(len(frames) in f):
                    isFight = 1
                    break
            frames.append({"name":index+"_"+str(len(frames)),"image":frame, "isFight":isFight,"isCCTV":isCCTV})
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    end = time.time()
    #print("SAVED:", video)
    print("TIME:", end - start)
    vc +=1
    if vc % 10 == 0:
        np.save('frames_1-'+str(vc), frames)
        print("SAVED:", str(vc))
