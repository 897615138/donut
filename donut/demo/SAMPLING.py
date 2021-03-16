def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    std_epsilon = 1e-4
    return z_mean + (z_log_var + std_epsilon) * epsilon


input_shape = (seq_len,)
intermediate_dim = 100
latent_dim = latent_dim

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
x = Dense(intermediate_dim, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var', activation='softplus')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# build decoder model
x = Dense(intermediate_dim, activation='relu', kernel_regularizer=regularizers.l2(0.001))(z)
x = Dense(intermediate_dim, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x_mean = Dense(seq_len, name='x_mean')(x)
x_log_var = Dense(seq_len, name='x_log_var', activation='softplus')(x)
outputs = Lambda(sampling, output_shape=(seq_len,), name='x')([x_mean, x_log_var])

vae = Model(inputs, outputs, name='vae_mlp')

# add loss
reconstruction_loss = mean_squared_error(inputs, outputs)
reconstruction_loss *= seq_len
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='adam')
