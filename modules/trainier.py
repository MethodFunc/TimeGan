import tensorflow as tf

import datetime
from .utils import train_embedding, train_supervisor, train_embedder, train_discriminator, train_generator, \
    get_discriminator_loss

from settings import logger

log = logger()


def embedding_train(embedder, recovery, mse, optims, real_series_iter, args, writer):
    for i in range(args.train_steps):
        st = datetime.datetime.now()
        X_ = next(real_series_iter)
        step_e_loss = train_embedding(X_, embedder, recovery, mse, optims['e0'])

        ed = datetime.datetime.now() - st

        log.info(f'[{i + 1}/{args.train_steps}] - {ed} | {step_e_loss:.6f}')

        with writer.as_default():
            tf.summary.scalar('Loss embdding Init', step_e_loss, step=i)


def supervisor_train(embedder, supervisor, generator, mse, optims, real_series_iter, random_series, args, writer):
    for i in range(args.train_steps):
        st = datetime.datetime.now()
        X_ = next(real_series_iter)
        Z_ = next(random_series)
        step_g_loss_s = train_supervisor(X_, Z_, embedder, supervisor, generator, mse, optims['gs'])
        ed = datetime.datetime.now() - st
        log.info(f'[{i + 1}/{args.train_steps}] - {ed} | {step_g_loss_s:.6f}')

        with writer.as_default():
            tf.summary.scalar('Loss generator Supervised Init', step_g_loss_s, step=i)


def train_loop(embedder, recovery, generator, discriminator, supervisor, mse, bce, optims, real_series_iter,
               random_series, args, writer):
    log.info('Training Auto Encoder')
    embedding_train(embedder, recovery, mse, optims, real_series_iter, args, writer)
    log.info('Auto Encoder Done.')
    log.info('Training Supervisor model')
    supervisor_train(embedder, supervisor, generator, mse, optims, real_series_iter, random_series, args, writer)
    log.info('Supervisor Done')

    log.info('Joint Training')
    step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = 0
    start_flag = 1
    for step in range(args.train_steps):
        if start_flag == 1 or step == 0:
            st = datetime.datetime.now()
            start_flag = 0

        # Train Genertor (twice as often as disdiscriminator)
        for kk in range(2):
            X_ = next(real_series_iter)
            Z_ = next(random_series)
            step_g_loss_u, step_g_loss_v, step_g_loss_s = train_generator(X_, Z_, embedder, supervisor, recovery,
                                                                          generator, discriminator, bce, mse,
                                                                          optims['g'])
            step_e_loss_t0 = train_embedder(X_, embedder, supervisor, recovery, mse, optims['e'])

        X_ = next(real_series_iter)
        Z_ = next(random_series)

        step_d_loss = get_discriminator_loss(X_, Z_, embedder, supervisor, generator, discriminator, bce, args.gamma)
        if step_d_loss > 0.15:
            step_d_loss = train_discriminator(X_, Z_, embedder, supervisor, generator, discriminator, optims['d'], bce,
                                              args.gamma)

        if step % 100 == 0:
            ed = datetime.datetime.now() - st
            start_flag = 1

            log.info(
                f"[{step:6.0f}/{args.train_steps}] - {ed} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}")

        with writer.as_default():
            tf.summary.scalar('G Loss S', step_g_loss_s, step=step)
            tf.summary.scalar('G Loss U', step_g_loss_u, step=step)
            tf.summary.scalar('G Loss V', step_g_loss_v, step=step)
            tf.summary.scalar('E Loss T0', step_e_loss_t0, step=step)
            tf.summary.scalar('D Loss', step_d_loss, step=step)


