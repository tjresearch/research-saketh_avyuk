import numpy as np
import matplotlib.pyplot as plt
from Pose_Process import process

test_frames = np.load("frames_1-10.npy")
raw_data = [(x['name'],x['image'],x['isFight']) for x in test_frames]
pose_to_fight = []
counter = 0
for n in raw_data[0::10]:
    canvas = process(n[1])
    print(canvas.shape)
    plt.imshow(canvas)
    plt.show()
    pose_to_fight.append({'name':n[0],'image':canvas,'isFight':n[2]})
    print(counter)
    counter+=1
    if counter % 100 == 0 and counter != 0:
        np.save('frames_1-10_input_data_v2', pose_to_fight)