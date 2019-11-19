
# coding: utf-8

# In[1]:


from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization, Flatten, LeakyReLU 
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, optimizers
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[2]:


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


# In[3]:


# print(image_paths)
# print(emotions)


# In[4]:


def path_to_tensor(img_path):
    return np.expand_dims(image.img_to_array(image.load_img(img_path, target_size=(224, 224))), axis=0)

def paths_to_tensor(img_paths):
    return np.vstack([path_to_tensor(img_path) for img_path in tqdm(img_paths)])

def get_file(path):
    return pd.read_csv(path).head()


# In[5]:


all_tensors = paths_to_tensor(image_paths).astype('float32')/255
all_targets = np_utils.to_categorical(emotions)


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(all_tensors, all_targets, test_size = 0.2, random_state = 7)


# In[7]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, padding='same', input_shape=(224,224,3)))
model.add(LeakyReLU(alpha = 0.05))
model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', input_shape=(224,224,3)))
model.add(LeakyReLU(alpha = 0.05))
model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(MaxPooling2D(pool_size=3))

model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))
model.add(Dense(NUM_EMOTIONS, activation='softmax'))

print(model.summary())
model_json = model.to_json()
with open("baselinemodel.json", "w") as json_file:
    json_file.write(model_json)

sgd = optimizers.SGD(lr=0.003)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


# In[ ]:


epochs = 500

tb = TensorBoard(log_dir='baseline_logs')
ckpt = ModelCheckpoint(filepath='baselineweightsfile.h5',
                               verbose=1, save_best_only=True)

history = model.fit(x_train, y_train,
          validation_split = 0.2, epochs=epochs, batch_size=500, callbacks=[ckpt, tb], verbose=1)


# In[ ]:


predictions = model.predict(y_test)

