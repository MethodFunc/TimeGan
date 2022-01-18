import pickle
import tensorflow as tf
import numpy as np
import datetime
from models.definemodel import call_models

from modules.dataloader import simply_loader
from modules.preprocessing import MinmaxScaler
from modules.location_module import location_preprocessing
from settings import arg_setting, default_setting, logger
from modules.trainier import train_loop


def make_random_data(x, y):
    while True:
        yield np.random.uniform(low=0, high=1, size=(x, y))


def main(model_dir, log_dir, args):
    log = logger()
    writer = tf.summary.create_file_writer(log_dir.as_posix())
    log.info('Data Loading')
    raw_data = simply_loader(args)

    log.info(f'{args.location} Data Proprocessing')
    pre_data = location_preprocessing(raw_data, args)

    scale = MinmaxScaler()

    scaled = scale.fit(pre_data.loc[args.date_start:args.date_end])

    # Scale Save
    with open(f'{model_dir}/scale.pickle', 'wb') as f:
        pickle.dump(scale, f)

    temp_data = [scaled[i:i + args.seq_len].values for i in range(len(scaled) - args.seq_len)]

    real_series = (tf.data.Dataset
                   .from_tensor_slices(temp_data)
                   .shuffle(buffer_size=args.seq_len)
                   .batch(args.batch_size))

    real_series_iter = iter(real_series.repeat())

    random_series = iter(tf.data.Dataset
                         .from_generator(make_random_data, output_types=tf.float32, args=(args.seq_len, args.n_seq))
                         .batch(args.batch_size)
                         .repeat()
                         )

    # Declare Models
    log.info('Declare models')
    embedder, recovery, generator, discriminator, supervisor = call_models(args)

    mse = tf.keras.losses.MeanSquaredError()
    bce = tf.keras.losses.BinaryCrossentropy()

    optims = {
        "e0": tf.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta, beta_2=args.beta_2),
        "gs": tf.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta, beta_2=args.beta_2),
        "e": tf.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta, beta_2=args.beta_2),
        "d": tf.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta, beta_2=args.beta_2),
        "g": tf.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta, beta_2=args.beta_2)
    }

    log.info('Start Training')
    total_st = datetime.datetime.now()
    train_loop(embedder, recovery, generator, discriminator, supervisor, mse, bce, optims, real_series_iter,
               random_series, args, writer)
    total_ed = datetime.datetime.now() - total_st

    log.info('End Training')

    log.info('Build Synthetic Models')
    input_layers = tf.keras.layers.Input(shape=(args.seq_len, args.n_seq), name='Input_data')
    x = generator(input_layers)
    x = supervisor(x)
    output_layers = recovery(x)
    synthetic_model = tf.keras.models.Model(inputs=input_layers, outputs=output_layers, name='Synthetic_Model')
    file_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'synthetic_{file_str}'
    synthetic_model.save(model_dir / f'{file_name}')
    synthetic_model.save_weights(model_dir / f'{file_name}_weight')
    log.info('Save Models and model_weight')
    log.info(f'Elapsed Time: {total_ed}')
    log.info(f'save model Directory: {str(model_dir)}')

    log.info('Args Setting List')
    for key, values in args._get_kwargs():
        log.info(f'{key}: {values}')


if __name__ == '__main__':
    args = arg_setting()
    log_dir, model_dir = default_setting()

    main(model_dir=model_dir, log_dir=log_dir, args=args)
