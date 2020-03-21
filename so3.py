import tensorflow as tf

def mat(x):
    # size: [*, 3] -> [*, 3, 3]
    x_ = tf.reshape(x, [-1, 3])
    x1, x2, x3 = tf.split(x_, 3, axis=-1)
    O = tf.zeros_like(x1)
    X = tf.stack([tf.stack([O, -x3, x2], axis=1),
                  tf.stack([x3, O, -x1], axis=1),
                  tf.stack([-x2, x1, O], axis=1)], axis=1)


    return tf.reshape(X, [-1, 3, 3])