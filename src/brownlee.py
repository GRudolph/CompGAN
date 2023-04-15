# example of a wgan for generating handwritten digits
import os
import numpy as np
import pandas as pd
from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint, max_norm
from matplotlib import pyplot

from keras.utils import plot_model, Progbar
import download_data as dl # download_data.py
import macro #macro.py
import preprocess as pre # preprocess.py
import evaluation as eval # evaluation.py

def load_data():
    
    if os.path.isfile(macro._LOCAL_SAVE_DATA)==0:

        # download data and compute featuers (see "download_data.py")
        # atomic_numbers use to compute composition vector
        # labels is target properties (formation energy)
        train_labels, compositions, features, atomic_numbers = dl.get_data()

        # compute bag-of-atom vector that trains GAN (see "preprocess.py")
        boa_vectors = pre.compute_bag_of_atom_vector(compositions, atomic_numbers)
        train_data = np.concatenate([boa_vectors,features],axis=1)

        save_data = pd.DataFrame(np.concatenate([train_labels,train_data],axis=1))
        save_data.to_csv(macro._LOCAL_SAVE_DATA, index=False, header=False)

    else:
        data = pd.read_csv(macro._LOCAL_SAVE_DATA,delimiter=',',engine="python",header=None)
        data = np.array(data)
        train_labels, train_data = np.split(data,[1],axis=1)

    # normalization of training data such that min is 0 and max is 1 (see "preprocess.py")
    normalized_train_data, data_max, data_min = pre.normalize_for_train(train_data)
    normalized_train_labels, max_train_prop, min_train_prop = pre.normalize_for_train(train_labels)

    # Save normalization parameter to .csv to use generation
    save_data = pd.DataFrame(np.concatenate([max_train_prop,min_train_prop,data_max,data_min],axis=0))
    save_data.to_csv(macro._SAVE_NORMALIZATION_PARAM, index=False, header=False)
    return train_data, train_labels

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

# define the standalone critic model
def define_critic(in_shape):
    model = Sequential(name = "Critic") 
    model.add(Dense(2*macro._LAYER_DIM, activation='relu', input_shape=(in_shape,), kernel_constraint=max_norm(0.5)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(macro._LAYER_DIM, activation='relu', kernel_constraint=max_norm(0.5)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='linear'))
    model.add(Flatten())
    # compile model
    opt = RMSprop(lr=0.00005)# Apply weight clipping to the discriminator weights
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# define the standalone generator model
def define_generator(data_shape, latent_dim):
    model = Sequential(name="Generator")
    # comes from expected shape(32,200)
    model.add(Dense(32, activation='relu', input_shape=(latent_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(2*macro._LAYER_DIM, activation='relu'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(data_shape, activation='tanh'))
    return model

# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
    # make weights in the critic not trainable
    for layer in critic.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect them
    model = Sequential(name="GAN")
    # add generator
    model.add(generator)
    # add the critic
    model.add(critic)
    # compile model
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# load images
def load_real_samples():
    # load dataset
    (trainX, trainy) = load_data()
    return trainX

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels, -1 for 'real'
    y = -ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels with 1.0 for 'fake'
    y = ones((n_samples, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(10 * 10):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
    # plot history
    pyplot.plot(d1_hist, label='crit_real')
    pyplot.plot(d2_hist, label='crit_fake')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    pyplot.savefig('plot_line_plot_loss.png')
    pyplot.close()

# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):

    # if you need progress bar
    progbar = Progbar(target=n_epochs)

    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # lists for keeping track of loss
    c1_hist, c2_hist, g_hist = list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        # update the critic more than the generator
        c1_tmp, c2_tmp = list(), list()
        for _ in range(n_critic):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update critic model weights
            c_loss1 = c_model.train_on_batch(X_real, y_real)
            c1_tmp.append(c_loss1)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update critic model weights
            c_loss2 = c_model.train_on_batch(X_fake, y_fake)
            c2_tmp.append(c_loss2)
        # store critic loss
        c1_hist.append(mean(c1_tmp))
        c2_hist.append(mean(c2_tmp))
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = -ones((n_batch, 1))
        # update the generator via the critic's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        g_hist.append(g_loss)
        # summarize loss on this batch
        #print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
        # evaluate the model performance every 'epoch'
        #if (i+1) % bat_per_epo == 0:
        #    summarize_performance(i, g_model, latent_dim)
    # line plots of loss
        progbar.add(1, values=[("c1",c1_hist[-1]), ("c2",c2_hist[-1]), ("gen_loss", g_loss)])
    #plot_history(c1_hist, c2_hist, g_hist)

# size of the latent space
latent_dim = 200

dataset = load_real_samples()
print(dataset.shape)

# create the critic
critic = define_critic(147)
# create the generator
generator = define_generator(147, latent_dim)
# create the gan
gan_model = define_gan(generator, critic)
gan_model.summary()
plot_model(gan_model, to_file='brownlee_model.png', show_shapes=True, show_layer_names=True)
# load image data
dataset = load_real_samples()
print(dataset.shape)
# train model
train(generator, critic, gan_model, dataset, latent_dim, n_epochs=macro._MAX_EPOCH, n_batch=256)