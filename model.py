from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, ConvLSTM2D, BatchNormalization, Flatten
import keras

#how this works--
#the functional api was used so it would be easier to utilize multiple inputs in the future if necessary
#other inputs could include some of the higher order relationships other papers have looked at (vector fields, crowd movement, etc.)
#primary input is going to be an image with a white background with estimated poses for all individuals on top of it -
#having them all represented within the original image is good because it preserves spatial relationships, whereas having just the poses
#represented as a matrix or something might not work super well.
#the change in that image also accounts for bystanders because the bystanders poses wouldn't change over time or if they did it wouldnt be like violent
#mess with filter size and kernel size


def get_model(timesteps,x,y,r=1,learning_rate=0.001):
    #r controls size of filters - should generalize to like how much space the violent poses actually take up in the image (may change this later)
    primary_input = Input(shape=(timesteps, x, y, 1))
    a = ConvLSTM2D(filters=256*r, kernel_size=(4, 4), input_shape=(timesteps, x, y, 1), return_sequences=True)(primary_input)
    a = Dropout(0.1)(a)
    a = BatchNormalization()(a)
    a = ConvLSTM2D(filters=256*r, kernel_size=(4, 4),
                   return_sequences=True)(a)
    a = Dropout(0.1)(a)
    a = BatchNormalization()(a)
    a = ConvLSTM2D(filters=64*r, kernel_size=(3, 3),
                   return_sequences=True)(a)
    a = Dropout(0.1)(a)
    a = BatchNormalization()(a)
    a = BatchNormalization()(a)
    a = ConvLSTM2D(filters=32*r, kernel_size=(2, 2),
                   return_sequences=True)(a)
    a = Flatten()(a)
    a = Dense(16)(a)
    a = Dense(8)(a)
    a = Dense(4)(a)
    a = Dense(2)(a)
    a = Dense(1)(a)
    model = Model(inputs=primary_input, outputs=a)

    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                amsgrad=False)  # clipvalue=2.0)#,clipnorm=1.0) #clipvalue=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model