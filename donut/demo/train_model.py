from tfsnippet.utils import get_variables_as_dict, VariableSaver
import tensorflow as tf


with tf.Session().as_default():
    # Train the model.

    # Remember to get the model variables after the birth of a
    # `predictor` or a `trainer`.  The :class:`Donut` instances
    # does not build the graph until :meth:`Donut.get_score` or
    # :meth:`Donut.get_training_loss` is called, which is
    # done in the `predictor` or the `trainer`.
    var_dict = get_variables_as_dict(model_vs)

    # save variables to `save_dir`
    saver = VariableSaver(var_dict, save_dir)
    saver.save()

with tf.Session().as_default():
    # Restore variables from `save_dir`.
    saver = VariableSaver(get_variables_as_dict(model_vs), save_dir)
    saver.restore()
