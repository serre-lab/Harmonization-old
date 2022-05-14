import tensorflow as tf
import numpy as np

def learning_rate_schedule_wrapper(training_steps_per_epoch, lr_schedule, base_learning_rate):
  """Wrapper around the learning rate schedule."""

  def learning_rate_schedule(current_epoch, current_batch):
    epoch = current_epoch + float(current_batch) / training_steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = lr_schedule[0]
    if epoch < warmup_end_epoch:
      # Learning rate increases linearly per step.
      return (base_learning_rate * warmup_lr_multiplier *
              epoch / warmup_end_epoch)
    for mult, start_epoch in lr_schedule:
      if epoch >= start_epoch:
        learning_rate = base_learning_rate * mult
      else:
        break
    return learning_rate
  return learning_rate_schedule


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  def __init__(self, schedule, model=None):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.epochs = -1
    self.prev_lr = -1

    if model is not None:
      self.model = model

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    lr = self.schedule(self.epochs, batch)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      tf.keras.backend.set_value(self.model.optimizer.lr, lr)
      self.prev_lr = lr