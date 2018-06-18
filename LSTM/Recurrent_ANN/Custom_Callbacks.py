import keras
import warnings
import numpy as np
from keras import backend as K
import STOP_TRAINING_REASON as tr

class Overfitting_callback(keras.callbacks.Callback):
    """Stop training when model overfits.

        # Arguments
            monitor: quantities to be monitored(list of at least two elements).
            patience: number of epochs with overfitting
                after which training will be stopped.
            verbose: verbosity mode.
            mode: one of {auto, min, max}. In `min` mode,
                training will stop when the quantity
                monitored has stopped decreasing; in `max`
                mode it will stop when the quantity
                monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred
                from the name of the monitored quantity.
        """

    def __init__(self, monitor=['loss', 'val_loss'],
                 patience=0, verbose=0, mode='auto'):
        super(Overfitting_callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Overfitting mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor[1]:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less


    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get(self.monitor[0])
        val_loss = logs.get(self.monitor[1])
        if val_loss is None:
            warnings.warn(
                'Overfitting conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        # train_loss > val_loss
        if not self.monitor_op(train_loss, val_loss):
            self.wait = 0

        # train_loss < val_loss
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True




    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            tr.overfitting=True
            print('Epoch %05d: Model overfit. Training stopped.' % (self.stopped_epoch + 1))

class LoadBestWeigtsReduceLR(keras.callbacks.Callback):
    """Loads curernt best weights and reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the best perfroming weights to date is loaded and learning rate is reduced.


    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
        weights_path: path to current best weights
    """

    def __init__(self, monitor='val_loss', factor=0.5, patience=10,
                 verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0,
                 weights_path="/tmp/weights.hdf5"):
        super(LoadBestWeigtsReduceLR, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.weights_path = weights_path
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0

            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        self.model.load_weights(self.weights_path)#Load best weights
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateauAndLoadBestWeights, loading weigths reducing learning'
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0

