import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import h5py

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose, Concatenate, Dropout, Add, Subtract, MaxPooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import metrics
from keras.utils import to_categorical

# Seed for Reproducibility
np.random.seed(31415)
epochs = 1000
batch_size = 32

# Turn on/off the training/testing workflow
train = False
test = True

# Define the Convolutional Block, for Easy
# weight Sharing
def ConvBlock(im_dim=150, im_ch=3):
    Im_Input = Input(shape=(im_dim, im_dim, im_ch))
    C1 = Conv2D(64, 3, padding='same', activation='relu')(Im_Input)
    C2 = Conv2D(64, 3, padding='same', activation='relu')(C1)
    MP1 = MaxPooling2D(2)(C2)
    MPDO1 = Dropout(0.25)(MP1)
    C3 = Conv2D(128, 3, padding='same', activation='relu')(MPDO1)
    C4 = Conv2D(128, 3, padding='same', activation='relu')(C3)
    MP2 = MaxPooling2D(2)(C4)
    MPDO2 = Dropout(0.25)(MP2)
    C4 = Conv2D(128, 3, padding='same', activation='relu')(MPDO2)
    C5 = Conv2D(128, 3, padding='same', activation='relu')(C4)
    MP3 = MaxPooling2D(2)(C5)
    MPDO3 = Dropout(0.25)(MP3)

    flat = Flatten()(MPDO3)

    H1 = Dense(128, activation='relu')(flat)
    H2 = Dense(256, activation='relu')(H1)
    H3 = Dense(256, activation='relu')(H2)

    return Model(Im_Input, H3)


# Define the method for loading the training or testing data
# It leverages the sklearn pairs library, 
def LoadData(im_dim=150, im_ch=3, method='train'):
    from sklearn.datasets import fetch_lfw_pairs
    import warnings

    start = 0
    stop = 250
    color = False

    if im_dim < 250:
        start = (250-im_dim)//2
        stop = start+im_dim
    if im_ch > 1:
        color = True

    warnings.filterwarnings("ignore")
    lfw_people = fetch_lfw_pairs(subset=method, 
                                 slice_=(slice(start, stop, None), 
                                 slice(start, stop, None)),
                                 resize=1, color=color)
    lfw_people.pairs = lfw_people.pairs/255.
    
    if method=='train':
        lfw_people.pairs = np.concatenate((lfw_people.pairs, np.swapaxes(np.array([lfw_people.pairs[:, 1, ...], lfw_people.pairs[:, 0, ...]]), 0, 1)), 0)
        lfw_people.target = np.concatenate((lfw_people.target, lfw_people.target), 0)

    return lfw_people


def abs_diff(X):
    s = X[0]
    for i in range(1, len(X)):
        s -= X[i]
    s = K.abs(s)
    return s

if __name__ == "__main__":
    # Specify some of the Properties
    im_dim = 150
    im_ch = 3

    # Initialize the Convolutional Subnetwork
    ConvSubNet = ConvBlock(im_dim, im_ch)

    # Specify the Image Inputs
    Im_In_1 = Input(shape=(im_dim, im_dim, im_ch))
    Im_In_2 = Input(shape=(im_dim, im_dim, im_ch))

    # Get the Feature Maps from the Conv Layers
    L1 = ConvSubNet(Im_In_1)
    L2 = ConvSubNet(Im_In_2)

    # Concatenate L1 and L2 to then Feed to Dense
    L = Subtract()([L1, L2])

    # Dense FF For Verification
    H1 = Dense(128, activation='relu')(L)
    DO1 = Dropout(0.25)(H1)
    H2 = Dense(256, activation='relu')(DO1)
    DO2 = Dropout(0.25)(H2)
    H3 = Dense(128, activation='relu')(DO2)
    H4 = Dense(256, activation='relu')(H3)
    DO3 = Dropout(0.25)(H4)
    H5 = Dense(128, activation='relu')(DO3)

    # Compute the Predicted Probability
    Output = Dense(2, activation='softmax')(H3)
    
    # Build the E2E CNN Model
    E2ECNN = Model([Im_In_1, Im_In_2], Output)
    E2ECNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    if train:
        print("Training...")
        # E2ECNN.load_weights("e2ecnn_weights_new.hdf5")
        train_data = LoadData()
        
        filepath="just_diff.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
        earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1)
        callback_list = [checkpoint, earlystop]

        E2ECNN.fit([train_data.pairs[:,0,...], train_data.pairs[:,1,...]], to_categorical(train_data.target, 2),
                    shuffle=True, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                    callbacks=callback_list, verbose=False)
    if test:
        print("Testing...")
        E2ECNN.load_weights("just_diff.hdf5")
        test_data = LoadData(method='test')
        print(E2ECNN.evaluate([test_data.pairs[:,0,...], test_data.pairs[:,1,...]], to_categorical(test_data.target)))
        # preds = E2ECNN.predict([test_data.pairs[:,0,...], test_data.pairs[:,1,...]])
        # preds_ = np.argmax(preds, axis=1)
        # print(preds_)
