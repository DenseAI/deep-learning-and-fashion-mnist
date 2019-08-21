import tensorflow as tf


def softmax_loss(y_true, y_pred):
    '''
    Softmax loss
    :param y_true:
    :param y_pred:
    :param s:
    :param e:
    :return:
    '''
    logits = y_pred
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    return loss
