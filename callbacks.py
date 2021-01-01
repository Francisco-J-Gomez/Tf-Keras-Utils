import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, \
    ReduceLROnPlateau
import math
import numpy as np
import io
from PIL import Image
import imageio
import matplotlib.pyplot as plt

# Tensorboard (https://keras.io/api/callbacks/tensorboard/)
logdir = "logs_directory"
tensorboard_callback = TensorBoard(logdir)
callbacks = [tensorboard_callback]
## In command line: %tensorboard --logdir logs

# Model checkpoint (https://keras.io/api/callbacks/model_checkpoint/)
callbacks = [ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1)]
callbacks = [ModelCheckpoint('saved_model', verbose=1)]
callbacks = [ModelCheckpoint('model.h5', verbose=1)]

# Early Stopping (https://keras.io/api/callbacks/early_stopping/)
callbacks = [EarlyStopping(patience=3, min_delta=0.05, baseline=0.8, mode='min', monitor='val_loss',
                           restore_best_weights=True, verbose=1)]

# CSV Logger (https://keras.io/api/callbacks/csv_logger/)
csv_file = 'training.csv'
callbacks = [CSVLogger(csv_file)]
# pd.read_csv(csv_file).head()


# Learning Rate Scheduler (https://keras.io/api/callbacks/learning_rate_scheduler/)
def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 1
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


callbacks = [LearningRateScheduler(step_decay, verbose=1), TensorBoard(log_dir=logdir)]

# ReduceLROnPlateau (https://keras.io/api/callbacks/reduce_lr_on_plateau/)
callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=1, min_lr=0.001),
           TensorBoard(log_dir=logdir)]

# Custom callbacks
# Common methods for training/testing/predicting:
# `on_(train|test|predict)_begin(self, logs=None)` Called at the beginning of `fit`/`evaluate`/`predict`.
# `on_(train|test|predict)_end(self, logs=None)`Called at the end of `fit`/`evaluate`/`predict`.
# `on_(train|test|predict)_batch_begin(self, batch, logs=None)`Called right before processing a batch during
# training/testing/predicting. Within this method, `logs` is a dict with `batch` and `size` available keys,
# representing the current batch number and the size of the batch.
# `on_(train|test|predict)_batch_end(self, batch, logs=None)`Called at the end of training/testing/predicting a batch.
# Within this method, `logs` is a dict containing the stateful metrics result.

# Training specific methods:
# `on_epoch_begin(self, epoch, logs=None)`Called at the beginning of an epoch during training.
# `on_epoch_end(self, epoch, logs=None)`Called at the end of an epoch during training.

callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs:
print("Epoch: {}, Val/Train loss ratio: {:.2f}".format(epoch, logs["val_loss"] / logs["loss"])))


class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.7):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        ratio = logs["val_loss"] / logs["loss"]
        print("Epoch: {}, Val/Train loss ratio: {:.2f}".format(epoch, ratio))

        if ratio > self.threshold:
            print("Stopping training...")
            self.model.stop_training = True


callbacks = [DetectOverfittingCallback()]

# Custom callback to visualize predictions
# Visualization utilities

plt.rc('font', size=20)
plt.rc('figure', figsize=(15, 3))


def display_digits(inputs, outputs, ground_truth, epoch, n=10):
    plt.clf()

    plt.yticks([])
    plt.grid(None)
    inputs = np.reshape(inputs, [n, 28, 28])
    inputs = np.swapaxes(inputs, 0, 1)
    inputs = np.reshape(inputs, [28, 28*n])
    plt.imshow(inputs)
    plt.xticks([28*x+14 for x in range(n)], outputs)
    for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if outputs[i] == ground_truth[i]:
            t.set_color('green')
        else:
            t.set_color('red')
    plt.grid(None)


GIF_PATH = 'train.gif'


class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, ground_truth, display_freq=10, n_samples=10):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n_samples = n_samples

    def on_epoch_end(self, epoch, logs=None):
        # Randomly sample data
        indexes = np.random.choice(len(self.inputs), size=self.n_samples)
        X_test, y_test = self.inputs[indexes], self.ground_truth[indexes]
        predictions = np.argmax(self.model.predict(X_test), axis=1)

        # Plot the digits
        display_digits(X_test, predictions, y_test, epoch, n=self.display_freq)

        # Save the figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))

        # Display the digits every 'display_freq' number of epochs
        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        imageio.mimsave(GIF_PATH, self.images, fps=1)

# callbacks=[VisCallback(x_test, y_test)]
