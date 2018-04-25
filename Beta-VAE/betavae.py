import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy.stats import norm

import h5py

import cv2

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
train = False
gen_data = False
selkey = 1
output_im = True

betas = [2, 4, 5, 10, 500]

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

# Define a method for loading in FEI data
def LoadFEIData():
    # Load in the Data
    main_images = h5py.File('main_images_1.h5', 'r')
    loaded_images = np.array(main_images['main_images'])
    return loaded_images


# Define the method for loading the training or testing data
# It leverages the sklearn pairs library, 
def LoadData(im_dim=150, im_ch=3, method='train', dup=True):
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
    
    if method=='train' and dup:
        lfw_people.pairs = np.concatenate((lfw_people.pairs, np.swapaxes(np.array([lfw_people.pairs[:, 1, ...], lfw_people.pairs[:, 0, ...]]), 0, 1)), 0)
        lfw_people.target = np.concatenate((lfw_people.target, lfw_people.target), 0)

    return lfw_people


if __name__ == "__main__":

    if output_im:
        random_emebds = np.random.normal(0,1,size=(120,32))
        # ims = LoadTrainData().images
        # random_ims = ims[np.random.choice(range(len(ims)), 10)]

        # for ii, random_im in enumerate(random_ims):
        #     plt.imshow(random_im)
        #     plt.axis('off')
        #     plt.savefig('final_figs/im_to_rep_{}.png'.format(ii))


    # Specify some of the Properties
    im_dim = 150
    im_ch = 3
    latent = 32
    features = 32
    # test_ims = LoadTrainData().images
    # test_ims_2 = LoadFEIData()
    
    # train_data = np.concatenate((test_ims, test_ims_2), axis=0)
    
    # print("Training data with shape {}".format(train_data.shape))
    
    for beta in betas:
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
        vae_loss = xent_loss + beta*K.mean(kl_loss)
        VAE.add_loss(vae_loss)
        
        VAE.compile(optimizer='adam', loss=None)
        
        VAE.load_weights("Completed_Training_beta={}.h5".format(beta))
        if gen_data:
            train_data = LoadData(dup=False)
            test_data = LoadData(method='test')

            VAE.load_weights("Checkpoint_Training_beta_double={}.hdf5".format(beta))

            header_str="id resp x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32"
            fmt_str= "%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f"

            train_1 = EncoderModel.predict(train_data.pairs[:,0,...])[selkey]
            train_1 = np.concatenate(((np.array(range(len(train_data.target)))+1).reshape(len(train_data.target),1), 
                                    train_data.target.reshape(len(train_data.target),1), train_1), 1)
            np.savetxt("betavae_{}_train_1.dat".format(beta), train_1, fmt=fmt_str, header=header_str, comments='')

            train_2 = EncoderModel.predict(train_data.pairs[:,1,...])[selkey]
            train_2 = np.concatenate(((np.array(range(len(train_data.target)))+1).reshape(len(train_data.target),1), 
                                    train_data.target.reshape(len(train_data.target),1), train_2), 1)
            np.savetxt("betavae_{}_train_2.dat".format(beta), train_2, fmt=fmt_str, header=header_str, comments='')

            test_1 = EncoderModel.predict(test_data.pairs[:,0,...])[selkey]
            test_1 = np.concatenate(((np.array(range(len(test_data.target)))+1).reshape(len(test_data.target),1), 
                                    test_data.target.reshape(len(test_data.target),1), test_1), 1)
            np.savetxt("betavae_{}_test_1.dat".format(beta), test_1, fmt=fmt_str, header=header_str, comments='')

            test_2 = EncoderModel.predict(test_data.pairs[:,1,...])[selkey]
            test_2 = np.concatenate(((np.array(range(len(test_data.target)))+1).reshape(len(test_data.target),1), 
                                    test_data.target.reshape(len(test_data.target),1), test_2), 1)
            np.savetxt("betavae_{}_test_2.dat".format(beta), test_2, fmt=fmt_str, header=header_str, comments='')

        # Fit Directly to the Data
        if train:
            print("Beginning training for beta={}.".format(beta))
            # VAE.load_weights("Completed_Training_beta={}.h5".format(beta))
            # train_data = LoadTrainData()
            filepath="Checkpoint_Training_beta_double={}.hdf5".format(beta)
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
            earlystop = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1)
            callback_list = [checkpoint, earlystop]

            VAE.fit(train_data, shuffle=True, verbose=0, epochs=100, batch_size=32, callbacks=callback_list)

            VAE.save_weights("Completed_Training_beta={}.h5".format(beta))
        
        if output_im:
            # decodeds = VAE.predict(random_ims)
            decodes = DecoderModel.predict(random_emebds)
            # plt.imshow(decodes[0])
            # plt.axis('off')
            # plt.savefig('final_figs/random/avg_{}.png'.format(beta))

            for ii, decoded in enumerate(decodes):
                plt.imshow(decoded)
                plt.axis('off')
                plt.savefig('final_figs/random/beta_vae_{}_rand_{}.png'.format(beta, ii))