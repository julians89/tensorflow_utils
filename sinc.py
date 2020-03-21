import tensorflow as tf

eps = 1e-6

def sinc1(x):
    """ sinc1: t -> sin(t)/t """
    x = tf.where(tf.abs(x) < eps, eps * tf.ones_like(x), x)
    return tf.sin(x) / x

def sinc2(x):
    """ sinc2: t -> (1 - cos(t)) / (t**2) """
    x = tf.where(tf.abs(x ** 2) < eps, eps * tf.ones_like(x), x ** 2)
    return (1. - tf.cos(x)) / x ** 2

def sinc3(x):
    """ sinc3: t -> (t - sin(t)) / (t**3) """
    x = tf.where(tf.abs(x ** 3) < eps, eps * tf.ones_like(x), x ** 3)
    return (x - tf.sin(x)) / x ** 3