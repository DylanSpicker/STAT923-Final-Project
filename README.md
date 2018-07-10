# STAT923 Final Project
The code within was created during a cumulative project for a Multivariate Analysis project. The idea was to attempt to learn low-dimensional latent representations of images of faces in order to conduct facial verification tasks. 

In general different autoencoder structures were used, with a focus on variational autoencoders that can be shifted to generative models.

The repository contains code for a standard Variational Autoencoder, the \beta-Variational Autoencoder (which helps learn independent latent representations), a VAE which is trained in batches and attempts to hold components constant for the purpose of learning disentangled representations, and a end-to-end CNN which attempts to do the classification without learning an explicit low-dimensional representation.
