import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import h5py

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import metrics

# Seed for Reproducibility
np.random.seed(31415)
latent = 32
epsilon_std = 1.0

# Re-paramerization trick to "sample" from the needed normal distribution.
# Sampling function from:
#  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# Define the 'encoder' network block;
#  Takes in image size; image channels; latent representation size
#  and returns a subnetwork that, using 2 Convolutional Layers, 1 hidden
#  layer, and a 'sampling' layer, returns the "z" latent representation.
def Encoder(im_dim=150, im_ch=3, latent=100, features=32):
    im_input = Input(shape=(im_dim, im_dim, im_ch))
    C1 = Conv2D(features, 3, padding='same', activation='relu')(im_input)
    C2 = Conv2D(features, 3, padding='same', activation='relu')(C1)
    flat = Flatten()(C2)
    H1 = Dense(120, activation='relu')(flat)

    z_mean = Dense(latent)(H1)
    z_log_sd = Dense(latent)(H1)

    z = Lambda(sampling)([z_mean, z_log_sd])

    return Model(im_input, [z, z_mean, z_log_sd])

# Define the 'decoder' network block;
#  Takes in image size; image channels; and latent representation size
#  and returns a subnework that, using a dense, reshape, and 2 transposed
#  convolutional layers returns the generated image.
def Decoder(im_dim=150, im_ch=3, latent=100, features=32):
    latent_input = Input(shape=(latent,))
    H1 = Dense(features*im_dim*im_dim)(latent_input)
    square = Reshape((im_dim, im_dim, features))(H1)
    C1 = Conv2DTranspose(features, 3, padding='same', activation='relu')(square)
    C2 = Conv2DTranspose(im_ch, 3, padding='same', activation='sigmoid')(C1)

    return Model(latent_input, C2)


# Define the method for loading the training data
# It leverages the sklearn pairs library, but then duplicates
# the pairs so as to allow for invariant training
def LoadTrainData(im_dim=150, im_ch=3):
    from sklearn.datasets import fetch_lfw_people
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
    lfw_people = fetch_lfw_people(min_faces_per_person=20, 
                                 slice_=(slice(start, stop, None), 
                                 slice(start, stop, None)),
                                 resize=1, color=color)
    lfw_people.images = lfw_people.images/255.
    return lfw_people

if __name__ == "__main__":
    # Specify some of the Properties
    im_dim = 150
    im_ch = 3
    latent = 32
    features = 32
    beta = 1
    
    print("Beginning training for beta={}.".format(beta))
    # Generate the Encoder and Decoder Modules
    EncoderModel = Encoder(im_dim, im_ch, latent, features)
    DecoderModel = Decoder(im_dim, im_ch, latent, features)

    # Model Input
    Im_In = Input(shape=(im_dim, im_dim, im_ch))
    Encoded, z_mean, z_log_sd = EncoderModel(Im_In)
    Im_Out = DecoderModel(Encoded)

    # Compile Model
    VAE = Model(Im_In, Im_Out)
    
    # Compute VAE loss
    xent_loss = im_dim * im_dim * metrics.binary_crossentropy(
        K.flatten(Im_In),
        K.flatten(Im_Out))
    kl_loss = - 0.5 * K.sum(1 + z_log_sd - K.square(z_mean) - K.exp(z_log_sd), axis=-1)
    vae_loss = K.mean(xent_loss + beta*kl_loss)
    VAE.add_loss(vae_loss)
    
    VAE.compile(optimizer='adam', loss=None)
    
    # Fit Directly to the Data
    train_data = LoadTrainData()

    filepath="standard_vae.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=1)
    callback_list = [checkpoint, earlystop]

    VAE.fit(train_data.images, shuffle=True, verbose=0, epochs=1000, batch_size=32, callbacks=callback_list)

    VAE.save_weights("standard_vae_final.h5".format(beta))
    print("Completed training for beta={}.".format(beta))