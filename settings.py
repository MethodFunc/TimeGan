import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
from pathlib import Path
import logging.config


def logger():
    logging.config.fileConfig('./logger.conf')
    return logging.getLogger('system')


def arg_setting():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.data_path = './data'
    args.gen_name = None
    args.location = 'westsouth'

    args.freq = '10T'

    # if date_start is None it'll mean start_index is 0
    args.date_start = '2021-04-01'
    # if date_end is None it'll call until the end columns
    args.date_end = None

    # Model Layers Setting
    args.sel_rnn = 'GRU'
    args.batchnorm = False

    args.hidden_dim = 128
    args.num_layers = 3
    args.seq_len = 36
    args.n_seq = 9

    args.batch_size = 128
    args.train_steps = 10
    args.gamma = 0.85

    args.lr = 0.0002
    args.beta = 0.5
    args.beta_2 = 0.999


    # Generator Setting
    args.model_path = './TimeGan_Result/experiment_03_model/synthetic_20220117_123226'
    args.scale_path = './TimeGan_Result/experiment_03_model/scale.pickle'

    return args


def default_setting():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    K.set_floatx('float32')

    result_path = Path('./TimeGan_Result')
    if not result_path.exists():
        result_path.mkdir()

    experiment = 1

    while True:
        log_dir = result_path / f'experiment_{experiment:02}'
        model_dir = result_path / f'experiment_{experiment:02}_model'

        if log_dir.exists() and log_dir.stat().st_size != 0:
            experiment += 1
        else:
            log_dir.mkdir(parents=True, exist_ok=True, mode=777)
            model_dir.mkdir(parents=True, exist_ok=True, mode=777)
            break

    return log_dir, model_dir
