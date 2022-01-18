import tensorflow as tf


@tf.function
def moment_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32) if y_true.dtype != tf.float32 else y_true
    y_pred = tf.cast(y_pred, tf.float32) if y_pred.dtype != tf.float32 else y_pred
    y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
    y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])

    g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
    g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
    return g_loss_mean + g_loss_var


@tf.function
def train_embedding(x, embedder, recovery, loss, optim):
    with tf.GradientTape() as tape:
        H = embedder(x)
        x_tilde = recovery(H)
        embedding_loss_t0 = loss(x, x_tilde)
        e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

    var_list = embedder.trainable_variables + recovery.trainable_variables

    gradients = tape.gradient(e_loss_0, var_list)
    optim.apply_gradients(zip(gradients, var_list))

    return tf.sqrt(embedding_loss_t0)


@tf.function
def train_supervisor(x, z, embedder, supervisor, generator, loss, optim):
    with tf.GradientTape() as tape:
        H = embedder(x)
        h_hat_supervised = supervisor(H)

        E_hat = generator(z)

        g_loss_s = loss(H[:, 1:, :], h_hat_supervised[:, :-1, :])

    var_list = supervisor.trainable_variables + generator.trainable_variables

    gradients = tape.gradient(g_loss_s, var_list)
    optim.apply_gradients(zip(gradients, var_list))

    return g_loss_s


@tf.function
def train_generator(X, Z, embedder, supervisor, recovery, generator, discriminator, loss_1, loss_2, optim):
    with tf.GradientTape() as tape:
        H = embedder(X)

        E_hat = generator(Z)
        H_hat = supervisor(E_hat)
        H_hat_supervisor = supervisor(H)

        Y_fake = discriminator(H_hat)
        Y_fake_e = discriminator(E_hat)

        X_hat = recovery(H_hat)

        g_loss_u = loss_1(tf.ones_like(Y_fake), Y_fake)
        g_loss_u_e = loss_1(tf.ones_like(Y_fake_e), Y_fake_e)
        g_loss_s = loss_2(H[:, 1:, :], H_hat_supervisor[:, :-1, :])

        g_loss_v = moment_loss(X, X_hat)

        generator_loss = (g_loss_u +
                          g_loss_u_e +
                          100 * tf.sqrt(g_loss_s) +
                          100 * g_loss_v)

        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        optim.apply_gradients(zip(gradients, var_list))

        return g_loss_u, g_loss_s, g_loss_v


@tf.function
def train_embedder(X, embedder, supervisor, recovery, loss_mse, optim):
    with tf.GradientTape() as tape:
        H = embedder(X)
        h_hat_supervised = supervisor(H)
        g_loss_s = loss_mse(H[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_tilde = recovery(H)
        e_loss_t0 = loss_mse(X, x_tilde)
        e_loss = 10 * tf.sqrt(e_loss_t0) + 0.1 * g_loss_s

    var_list = embedder.trainable_variables + recovery.trainable_variables
    gradients = tape.gradient(e_loss, var_list)
    optim.apply_gradients(zip(gradients, var_list))

    return tf.sqrt(e_loss_t0)


@tf.function
def get_discriminator_loss(X, Z, embedder, supervisor, generator, discriminator, loss, gamma):
    H = embedder(X)

    E_hat = generator(Z)
    H_hat = supervisor(E_hat)

    Y_fake = discriminator(H_hat)
    Y_fake_e = discriminator(E_hat)
    Y_real = discriminator(H)

    D_loss_real = loss(y_true=tf.ones_like(Y_real), y_pred=Y_real)
    D_loss_fake = loss(y_true=tf.zeros_like(Y_fake), y_pred=Y_fake)
    D_loss_fake_e = loss(y_true=tf.zeros_like(Y_fake_e), y_pred=Y_fake_e)

    D_loss = (D_loss_real + D_loss_fake + gamma * D_loss_fake_e)

    return D_loss


@tf.function
def train_discriminator(X, Z, embedder, supervisor, generator, discriminator, optim, loss, gamma):
    with tf.GradientTape() as tape:
        d_loss = get_discriminator_loss(X, Z, embedder, supervisor, generator, discriminator, loss, gamma)

    var_list = discriminator.trainable_variables
    gradients = tape.gradient(d_loss, var_list)
    optim.apply_gradients(zip(gradients, var_list))

    return d_loss
