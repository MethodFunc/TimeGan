import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Input, Activation, BatchNormalization, LSTM


class DefineModel:
    def __init__(self, args):
        self.n_seq = args.n_seq
        self.num_layer = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.bn = args.batchnorm
        self.sel_rnn = args.sel_rnn

    def make_rnn(self, output_dim, ac_func, name):
        model = Sequential(name=name)

        for i in range(self.num_layer):
            if self.sel_rnn == 'GRU':
                model.add(GRU(units=self.hidden_dim, return_sequences=True, name=f'GRU_{i + 1}'))
            elif self.sel_rnn == 'LSTM':
                model.add(LSTM(units=self.hidden_dim, return_sequences=True, name=f'LSTM_{i + 1}'))
            else:
                raise f'Do not support rnn func'

        model.add(Dense(units=output_dim))
        if self.bn:
            model.add(BatchNormalization())
        if ac_func == 'sigmoid':
            model.add(Activation(tf.nn.sigmoid))
        elif ac_func is None:
            pass

        return model

    def embedder(self):
        return self.make_rnn(self.hidden_dim, 'sigmoid', 'Embedder')

    def recovery(self):
        return self.make_rnn(self.n_seq, 'sigmoid', 'Recovery')

    def generator(self):
        return self.make_rnn(self.hidden_dim, 'sigmoid', 'Generator')

    def supervisor(self):
        return self.make_rnn(self.hidden_dim, 'sigmoid', 'Supervisor')

    def discriminator(self):
        return self.make_rnn(1, None, 'Discriminator')


def call_models(args):
    declare = DefineModel(args)

    embedder = declare.embedder()
    recovery = declare.recovery()
    generator = declare.generator()
    discriminator = declare.discriminator()
    supervisor = declare.supervisor()

    return embedder, recovery, generator, discriminator, supervisor


def build_model(embedder, recovery, generator, discriminator, supervisor, args):
    X = Input(shape=[args.seq_len, args.n_seq], name='RealData')
    Z = Input(shape=[args.seq_len, args.n_seq], name='RandomData')

    # Autoencode
    H = embedder(X)
    X_tilde = recovery(H)

    # Generator
    E_hat = generator(Z)
    H_hat = supervisor(E_hat)
    H_hat_supervicor = supervisor(H)

    # SyntheticData
    X_hat = recovery(H_hat)

    # Discriminator
    Y_fake = discriminator(H_hat)
    Y_real = discriminator(H)
    Y_fake_e = discriminator(E_hat)

    model_ae = Model(inputs=X, outputs=X_tilde, name='AutoEncoder')
    model_ads = Model(inputs=Z, outputs=Y_fake, name='AdversarialNet_Supervisor')
    model_ade = Model(inputs=Z, outputs=Y_fake_e, name='AdversarialNet')

    model_syn = Model(inputs=Z, outputs=X_hat, name='Synthetic Model')
    model_disc = Model(inputs=X, outputs=Y_real, name='Discriminator')

    return model_ae, model_ads, model_ade, model_syn, model_disc
