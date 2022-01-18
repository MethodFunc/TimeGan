import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from settings import arg_setting


def make_random_data(x, y):
    while True:
        yield np.random.uniform(low=0, high=1, size=(x, y))


print('Generating Data...')
args = arg_setting()

random_data = iter(
    tf.data.Dataset.from_generator(make_random_data, output_types=tf.float32, args=(args.seq_len, args.n_seq))
    .batch(args.batch_size)
    .repeat()
)

generator_model = tf.keras.models.load_model(args.model_path)

with open(args.scale_path, 'rb') as f:
    scale = pickle.load(f)

n_windows = 1440
generated_data = []

for i in range(int(n_windows/args.batch_size)):
    Z_ = next(random_data)
    data = generator_model(Z_)
    generated_data.append(data)

generated_data = np.array(np.vstack(generated_data))

inverse_data = scale.inverse(generated_data)
file_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_df = pd.DataFrame(inverse_data.reshape(-1, args.n_seq))

print('Save Synthetic Data...')

save_df.to_csv(f'./TimeGan_{file_str}.csv')

print('Generator Done.')