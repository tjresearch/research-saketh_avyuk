
# coding: utf-8

# In[ ]:


from keras.models import model_from_json
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

legend = pd.read_csv("legend.csv")#.head(125)
image_paths = legend["image"].values
for i in range(len(image_paths)):
    image_paths[i] = 'images/'+image_paths[i]
    
emotions = legend["emotion"].values
emotions = [emotion.lower() for emotion in emotions]

emotions_set = set(emotions)
NUM_EMOTIONS = len(emotions_set)
emotions_to_int = {}
int_to_emotions = {}
for pair in list(enumerate(emotions_set)):
    emotions_to_int[pair[1]] = pair[0]
    int_to_emotions[pair[0]] = pair[1]
emotions = np.array([emotions_to_int[emotion] for emotion in emotions])


def path_to_tensor(img_path):
    return np.expand_dims(image.img_to_array(image.load_img(img_path, target_size=(224, 224))), axis=0)

def paths_to_tensor(img_paths):
    return np.vstack([path_to_tensor(img_path) for img_path in tqdm(img_paths)])

def get_file(path):
    return pd.read_csv(path).head()


all_tensors = paths_to_tensor(image_paths).astype('float32')/255
all_targets = np_utils.to_categorical(emotions)

x_train, x_test, y_train, y_test = train_test_split(all_tensors, all_targets, test_size = 0.2, random_state = 7)

json_file = open('baselinemodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("baselineweightsfile.h5")

predictions = model.predict(x_test)

