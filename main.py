import cv2
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=(8,8), activation='relu', input_shape=(128,72,3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
#model.add(Conv2D(128, kernel_size=(8,8), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(8,8), activation='relu'))
model.add(Flatten())
#model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1, activation='softmax'))
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

raw_input_data = np.load("frames_1-10_input_data.npy")
print(raw_input_data.shape)


#pca = PCA(n_components=2, svd_solver='full')


X = []
y = []
for sample in raw_input_data:


    scale_percent = 20  # percent of original size
    width = int(360 * scale_percent / 100)
    height = int(640 * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(sample['image'], dim, interpolation=cv2.INTER_AREA)
    X.append(resized)

    y.append(sample['isFight'])
X = np.array(X)
y = np.array(y)

# test_size = 0.23, random_state=4965

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4965)
print(X_train.shape)
model.fit(X_train, y_train,
          batch_size=8,
          epochs=50)
print(model.evaluate(X_test,y_test))