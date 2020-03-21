import tensorflow as tf

def power_iteration(A, max_steps=10):
    """
    Takes a square matrix A [-1, N, N] and finds it's dominant eigenvalue / eigenvector pair
    :param A:
    :return:
    """

    def cond(r, i):
        return i < max_steps

    def body(r, i):
        i += 1
        r = tf.nn.l2_normalize(tf.reshape(tf.matmul(A, tf.expand_dims(r, -1)), [-1, N]), axis=-1)
        return [r, i]

    batch_size = tf.shape(A)[0]
    N = tf.shape(A)[1]

    # r0 = tf.random.uniform(shape=[batch_size, N], dtype=tf.float32)
    # TODO: allow negative values here?
    r0 = tf.random.uniform(shape=[batch_size, N], minval=-1, maxval=1, dtype=tf.float32)
    r0 = tf.nn.l2_normalize(r0, axis=-1)
    i0 = tf.constant(0, dtype=tf.int32)

    r_final, a = tf.while_loop(cond, body, loop_vars=[r0, i0], back_prop=True, parallel_iterations=10)

    return r_final

