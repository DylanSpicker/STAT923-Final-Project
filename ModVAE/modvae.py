import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy.stats import norm

import h5py

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose, Average, Multiply, Add, Subtract
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import metrics

# Seed for Reproducibility
np.random.seed(31415)

# Training Paremeters
latent=32
epsilon_std=1.0

gen_data = False
train = False
output_im = True 
selkey = 1

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

# A function which batches the Brazil Data;
#   For now we will just take the first 3 individuals, and generate
#   the 11 rotational, and 11 fixed rotation sets, for testing purposes
def CreateBatches():
    # Load in the Data
    main_images = h5py.File('main_images_1.h5', 'r')
    loaded_images = np.array(main_images['main_images'])

    # batch_1_idx = list(range(11))
    # batch_2_idx = list(range(15,26))
    # batch_3_idx = list(range(29,40))
    # batch_4_idx = list(range(0,141,14))
    # batch_5_idx = list(range(1,142,14))
    # batch_6_idx = list(range(2,143,14))
    filter_base = np.zeros(latent, dtype=np.uint8)

    filter_r = filter_base
    filter_r[0] = 1
    filter_i = filter_base + 1
    filter_i[0] = 0
    
    # Generate 600 Random Poses
    pose = np.random.choice(range(11), (600,1))
    people = np.array([np.random.choice(range(200), 11, replace=False) for _ in range(600)])

    # Random Indices
    random_idx = people*14+pose
    return loaded_images[list(range(0,2800,14))+list(random_idx[:,0])],\
           loaded_images[list(range(1,2800,14))+list(random_idx[:,1])],\
           loaded_images[list(range(2,2800,14))+list(random_idx[:,2])],\
           loaded_images[list(range(3,2800,14))+list(random_idx[:,3])],\
           loaded_images[list(range(4,2800,14))+list(random_idx[:,4])],\
           loaded_images[list(range(5,2800,14))+list(random_idx[:,5])],\
           loaded_images[list(range(6,2800,14))+list(random_idx[:,6])],\
           loaded_images[list(range(7,2800,14))+list(random_idx[:,7])],\
           loaded_images[list(range(8,2800,14))+list(random_idx[:,8])],\
           loaded_images[list(range(9,2800,14))+list(random_idx[:,9])],\
           loaded_images[list(range(10,2800,14))+list(random_idx[:,10])],\
           np.array([filter_r]*200+[filter_i]*600)


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

    if output_im:
        random_emebds = np.zeros(shape=(32,1))
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
    features = 16
    batch_size = 11

    alpha = 1
    beta = 10
    lambda_ = 0.01

    ##########################################################
    ### Model Construction Phase: 
    ####  Build a model which takes in each item in the batch
    ####  and returns images correspondingly
    ##########################################################
    # Generate the Encoder and Decoder Modules
    EncoderModel = Encoder(im_dim, im_ch, latent, features)
    DecoderModel = Decoder(im_dim, im_ch, latent, features)

    # Create the List for Auto-Generated Layers
    Model_Inputs = []
    Encoded_Out = []
    Z_mean = []
    Z_log_sd = []
    Clamped_Latents = []
    Model_Outputs = []

    # Filter Input
    Filter_In = Input(shape=(latent,))
    Unit_Con = Input(shape=(latent,))
    Filter_P = Subtract()([Unit_Con, Filter_In])

    # Generate the Correct Input -> Encoder Through for each Batch Item
    for _ in range(batch_size):
        Model_Inputs.append(Input(shape=(im_dim, im_dim, im_ch)))
        z, z_mean, z_log_sd = EncoderModel(Model_Inputs[-1])
        Encoded_Out.append(z)
        Z_mean.append(z_mean)
        Z_log_sd.append(z_log_sd)

    # Create Average and Filtered Average Layers
    E_mu = Average()(Encoded_Out)
    F_mu = Multiply()([E_mu, Filter_P])

    # Generate the Correct Latent -> Decoder for each Batch Item
    for ii in range(batch_size):
        Clamped_Latents.append(Add()([Multiply()([Encoded_Out[ii], Filter_In]), F_mu]))
        Model_Outputs.append(DecoderModel(Clamped_Latents[-1]))

    # Add the Filter and Constant as Inputs to Model
    Model_Inputs.append(Filter_In)
    Model_Inputs.append(Unit_Con)

    # Produce the Model
    ModVAE = Model(Model_Inputs, Model_Outputs)
    
    KL_loss = 0
    RE_loss = 0
    MSE_loss = 0

    # Add the Loss Function
    for jj in range(batch_size):
        KL_loss += -0.5 * K.sum(1 + Z_log_sd[jj] - K.square(Z_mean[jj]) - K.exp(Z_log_sd[jj]), axis=-1)
        RE_loss += im_dim*im_dim*metrics.binary_crossentropy(K.flatten(Model_Inputs[jj]), 
                                                               K.flatten(Model_Outputs[jj]))
        MSE_loss += metrics.mean_squared_error(F_mu, Clamped_Latents[jj])

    vae_loss = K.mean(alpha*RE_loss + beta*KL_loss + lambda_*MSE_loss)
    ModVAE.add_loss(vae_loss)

    ModVAE.compile(optimizer='adam', loss=None)
    ModVAE.summary()

    # ModVAE.load_weights("modvae_32_final.h5")

    if gen_data:
        train_data = LoadData(dup=False)
        test_data = LoadData(method='test')

        ModVAE.load_weights("modvae_32_final.h5")
        header_str="id resp x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32"
        fmt_str= "%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f"

        train_1 = EncoderModel.predict(train_data.pairs[:,0,...])[selkey]
        train_1 = np.concatenate(((np.array(range(len(train_data.target)))+1).reshape(len(train_data.target),1), 
                                   train_data.target.reshape(len(train_data.target),1), train_1), 1)
        np.savetxt("modvae_train_1.dat", train_1, fmt=fmt_str, header=header_str, comments='')

        train_2 = EncoderModel.predict(train_data.pairs[:,1,...])[selkey]
        train_2 = np.concatenate(((np.array(range(len(train_data.target)))+1).reshape(len(train_data.target),1), 
                                   train_data.target.reshape(len(train_data.target),1), train_2), 1)
        np.savetxt("modvae_train_2.dat", train_2, fmt=fmt_str, header=header_str, comments='')

        test_1 = EncoderModel.predict(test_data.pairs[:,0,...])[selkey]
        test_1 = np.concatenate(((np.array(range(len(test_data.target)))+1).reshape(len(test_data.target),1), 
                                  test_data.target.reshape(len(test_data.target),1), test_1), 1)
        np.savetxt("modvae_test_1.dat", test_1, fmt=fmt_str, header=header_str, comments='')

        test_2 = EncoderModel.predict(test_data.pairs[:,1,...])[selkey]
        test_2 = np.concatenate(((np.array(range(len(test_data.target)))+1).reshape(len(test_data.target),1), 
                                  test_data.target.reshape(len(test_data.target),1), test_2), 1)
        np.savetxt("modvae_test_2.dat", test_2, fmt=fmt_str, header=header_str, comments='')

    if train:
        # print(ModVAE.inputs)
        # Filter through the Training Data
        filepath="modvae_32.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
        earlystop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=1)
        callback_list = [checkpoint, earlystop]

        im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, filters = CreateBatches()
        ModVAE.fit([im_1, 
                    im_2, 
                    im_3, 
                    im_4, 
                    im_5, 
                    im_6, 
                    im_7, 
                    im_8, 
                    im_9, 
                    im_10, 
                    im_11, 
                    filters, 
                    np.ones(shape=filters.shape)-filters], epochs=1000, verbose=1, shuffle=True, callbacks=callback_list)
        
        ModVAE.save_weights("modvae_32_final.h5")
    
    if output_im:
        encoders = EncoderModel.predict(random_ims)[1]
        decodeds = DecoderModel.predict(encoders)
        
        for ii, decoded in enumerate(decodeds):
            plt.imshow(decoded)
            plt.axis('off')
            plt.savefig('final_figs/modvae_{}.png'.format(ii))