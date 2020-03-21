import tensorflow as tf
import so3
import sinc
import eig

def exp(x):
    x_ = tf.reshape(x, [-1, 6])
    #w, v = tf.split(tf.reshape(x_, [-1, 3, 2]), 2, axis=-1)

    #w are first 3
    w_idx = tf.reshape(tf.range(tf.shape(x_)[0]) * 6, [-1, 1]) + tf.reshape(tf.range(3), [1, -1])
    w = tf.gather(tf.reshape(x_, [-1]), w_idx)
    w = tf.reshape(w, [-1, 3])

    v_idx = tf.reshape(tf.range(tf.shape(x_)[0]) * 6, [-1, 1]) + tf.reshape(tf.range(3) + 3, [1, -1])
    v = tf.gather(tf.reshape(x_, [-1]), v_idx)
    v = tf.reshape(v, [-1, 3])

    t = tf.reshape(tf.norm(w, axis=1), [-1, 1, 1])
    W = so3.mat(w)
    S = tf.matmul(W, W)
    I = tf.eye(3, dtype=w.dtype)

    # Rodrigues' rotation formula.
    #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    #  = eye(3) + sinc1(t)*W + sinc2(t)*S
    R = I + sinc.sinc1(t)*W + sinc.sinc2(t)*S

    #V = sinc1(t)*eye(3) + sinc2(t)*W + sinc3(t)*(w*w')
    #  = eye(3) + sinc2(t)*W + sinc3(t)*S
    V = I + sinc.sinc2(t)*W + sinc.sinc3(t)*S

    p = tf.matmul(V, tf.reshape(v, [-1, 3, 1]))

    z = tf.tile(tf.reshape(tf.constant([0., 0., 0., 1.], dtype=x.dtype), [1, 1, 4]), [tf.shape(x_)[0], 1, 1])

    Rp = tf.concat([R, p], axis=2)
    g = tf.concat([Rp, z], axis=1)

    return tf.reshape(g, [-1, 4, 4])

def transform(g, a):
    a_ = tf.concat([a, tf.reduce_mean(tf.fill(tf.shape(a), 1.0), axis=-1, keepdims=True)], axis=-1)

    #b = tf.matmul(a_, tf.linalg.matrix_transpose(g))

    b = tf.cond(tf.equal(tf.rank(g), tf.rank(a)),
                lambda: tf.matmul(a_, tf.linalg.matrix_transpose(g)),
                lambda: tf.squeeze(tf.matmul(tf.expand_dims(a_, axis=-2), tf.linalg.matrix_transpose(g)), axis=-2))


    #b = tf.squeeze(tf.matmul(tf.expand_dims(a_, axis=-2), tf.linalg.matrix_transpose(g)), axis=-2)

    b0, b1, b2, _ = tf.split(b, 4, axis=-1)
    b = tf.concat([b0, b1, b2], axis=-1)

    return b

def decompose(g):

    batch_size = tf.shape(g)[0]
    g = tf.reshape(g, [-1, 4, 4])

    rot_mat_idx = tf.reshape(tf.range(batch_size) * 16, [batch_size, 1, 1]) + tf.reshape(tf.range(3) * 4, [1, 3, 1]) + tf.reshape(tf.range(3), [1, 1, 3])

    rot_mat = tf.gather(tf.reshape(g, [-1]), rot_mat_idx)
    rot_mat = tf.reshape(rot_mat, [-1, 3, 3])

    translation_idx = tf.reshape(tf.range(batch_size) * 16, [batch_size, 1]) + tf.reshape(tf.range(3) * 4 + 3, [1, 3])

    trans_vector = tf.gather(tf.reshape(g, [-1]), translation_idx)
    trans_vector = tf.reshape(trans_vector, [-1, 3])

    return rot_mat, trans_vector

def compose(R, t):

    batch_size = tf.shape(R)[0]


    rot_mat_idx = tf.reshape(tf.range(batch_size) * 16, [batch_size, 1, 1]) + tf.reshape(tf.range(3) * 4, [1, 3, 1]) + tf.reshape(tf.range(3), [1, 1, 3])
    rot_mat = tf.scatter_nd(tf.reshape(rot_mat_idx, [-1, 1]), tf.reshape(R, [-1]), [batch_size * 16])
    rot_mat = tf.reshape(rot_mat, [-1, 4, 4])


    translation_idx = tf.reshape(tf.range(batch_size) * 16, [batch_size, 1]) + tf.reshape(tf.range(3) * 4 + 3, [1, 3])
    #trans_mat = tf.scatter_nd(tf.reshape(translation_idx, [-1, 1]), tf.reshape(t, [-1]), [batch_size * 16])
    #trans_mat = tf.reshape(trans_mat, [-1, 4, 4])

    valid_rot = tf.scatter_nd(tf.reshape(rot_mat_idx, [-1, 1]), tf.ones([batch_size * 9]), [batch_size * 16])
    valid_trans = tf.scatter_nd(tf.reshape(translation_idx, [-1, 1]), tf.ones([batch_size * 3]), [batch_size * 16])
    invalid = 1. - (valid_rot + valid_trans)
    invalid = tf.reshape(invalid, [-1, 4, 4])

    mat = add_t(rot_mat, t)

    return mat + (tf.expand_dims(tf.eye(4), axis=0) * invalid)

#TODO: untested!!!!
def transform_matrix_around_arbitrary_point(g, t):
    """

    :param g: [-1, 4, 4] homogenous transformation matrix
    :param t: [-1, 3] point offset
    :return:

    Formula is T_g = t*T*(-t)
    """

    orig_shape = tf.shape(g)

    g = tf.reshape(g, [-1, 4, 4])

    t = tf.reshape(t, [-1, 3])

    batch_size = tf.shape(g)[0]

    idx_ = tf.reshape(tf.range(3) * 4 + 3, [1, -1])
    idx_ += tf.reshape(tf.range(batch_size) * 16, [-1, 1])

    t_mat = tf.scatter_nd(tf.reshape(idx_, [-1, 1]), tf.reshape(t, [-1]), shape=[batch_size * 16])
    t_mat = tf.reshape(t_mat, [batch_size, 4, 4])

    eye = tf.expand_dims(tf.eye(4), axis=0)

    return tf.reshape(tf.matmul(tf.matmul((t_mat + eye), g), (eye - t_mat)), orig_shape)

def add_t(g, t):
    """

    :param g: [-1, 4, 4]
    :param t: [-1, 3] tranlation vector
    :return:
    """
    batch_size = tf.shape(g)[0]

    idx_ = tf.reshape(tf.range(3) * 4 + 3, [1, -1])
    idx_ += tf.reshape(tf.range(batch_size) * 16, [-1, 1])

    t_mat = tf.scatter_nd(tf.reshape(idx_, [-1, 1]), tf.reshape(t, [-1]), shape=[batch_size * 16])
    t_mat = tf.reshape(t_mat, [batch_size, 4, 4])
    t_mat += tf.expand_dims(tf.eye(4), axis=0)
    return tf.matmul(g, t_mat)

def inverse(g):
    """
    :param g: [-1, 4, 4]
    :return:

    [[R:3x3, d:3x1],
     [0, 0, 0, 1]]

     inverse is:
     [[R^T, -R^T*d],
      [0, 0, 0, 1]]
    """

    orig_shape = tf.shape(g)

    g = tf.reshape(g, [-1, 4, 4])

    batch_size = tf.shape(g)[0]

    rot_mat_idx = tf.reshape(tf.range(batch_size) * 16, [batch_size, 1, 1]) + tf.reshape(tf.range(3) * 4, [1, 3, 1]) + tf.reshape(tf.range(3), [1, 1, 3])

    rot_mat = tf.gather(tf.reshape(g, [-1]), rot_mat_idx)
    rot_mat = tf.reshape(rot_mat, [-1, 3, 3])

    translation_idx = tf.reshape(tf.range(batch_size) * 16, [batch_size, 1]) + tf.reshape(tf.range(3) * 4 + 3, [1, 3])

    trans_vector = tf.gather(tf.reshape(g, [-1]), translation_idx)
    trans_vector = tf.reshape(trans_vector, [-1, 3, 1])


    rot_mat_transpose = tf.linalg.matrix_transpose(rot_mat)
    trans_vector_inv = tf.matmul(-rot_mat_transpose, trans_vector)


    rot_mat_transpose = tf.scatter_nd(tf.reshape(rot_mat_idx, [-1, 1]), tf.reshape(rot_mat_transpose, [-1]), [batch_size * 16])
    rot_mat_transpose = tf.reshape(rot_mat_transpose, [-1, 4, 4])

    trans_vector_inv = tf.scatter_nd(tf.reshape(translation_idx, [-1, 1]), tf.reshape(trans_vector_inv, [-1]), [batch_size * 16])
    trans_vector_inv = tf.reshape(trans_vector_inv, [-1, 4, 4])

    valid_rot = tf.scatter_nd(tf.reshape(rot_mat_idx, [-1, 1]), tf.ones([batch_size * 9]), [batch_size * 16])
    valid_trans = tf.scatter_nd(tf.reshape(translation_idx, [-1, 1]), tf.ones([batch_size * 3]), [batch_size * 16])
    invalid = 1. - (valid_rot + valid_trans)
    invalid = tf.reshape(invalid, [-1, 4, 4])

    ret = (rot_mat_transpose + trans_vector_inv) + (tf.expand_dims(tf.eye(4), axis=0) * invalid)


    return tf.reshape(ret, orig_shape)

def so3_to_unit_quaternion(g, epsilon=1.e-8):

    g_flat = tf.reshape(g, [-1])

    batch_size = tf.shape(g)[0]

    r_xx_idx = tf.range(batch_size) * 9
    R_xx = tf.gather(g_flat, r_xx_idx)

    r_xy_idx = tf.range(batch_size) * 9 + 1
    R_xy = tf.gather(g_flat, r_xy_idx)

    r_xz_idx = tf.range(batch_size) * 9 + 2
    R_xz = tf.gather(g_flat, r_xz_idx)

    r_yx_idx = tf.range(batch_size) * 9 + 3
    R_yx = tf.gather(g_flat, r_yx_idx)

    r_yy_idx = tf.range(batch_size) * 9 + 4
    R_yy = tf.gather(g_flat, r_yy_idx)

    r_yz_idx = tf.range(batch_size) * 9 + 5
    R_yz = tf.gather(g_flat, r_yz_idx)

    r_zx_idx = tf.range(batch_size) * 9 + 6
    R_zx = tf.gather(g_flat, r_zx_idx)

    r_zy_idx = tf.range(batch_size) * 9 + 7
    R_zy = tf.gather(g_flat, r_zy_idx)

    r_zz_idx = tf.range(batch_size) * 9 + 8
    R_zz = tf.gather(g_flat, r_zz_idx)


    #g_flat = tf.reshape(g, [-1, 9])
    #R_xx, R_xy, R_xz, R_yx, R_yy, R_yz, R_zx, R_zy, R_zz = tf.split(g_flat, 9, axis=-1)


    tr = R_xx + R_yy + R_zz

    s_0 = tf.sqrt(tf.maximum(tr+1.0, 1e-6)) * 2.
    qw_0 = 0.25 * s_0
    qx_0 = (R_zy - R_yz) / s_0
    qy_0 = (R_xz - R_zx) / s_0
    qz_0 = (R_yx - R_xy) / s_0

    q_0 = tf.concat([tf.expand_dims(qw_0, axis=-1), tf.expand_dims(qx_0, axis=-1), tf.expand_dims(qy_0, axis=-1), tf.expand_dims(qz_0, axis=-1)], axis=-1)

    s_1 = tf.sqrt(tf.maximum(1.0 + R_xx - R_yy - R_zz, 1e-6)) * 2  # // S=4 * qx
    qw_1 = (R_zy - R_yz) / s_1
    qx_1 = 0.25 * s_1
    qy_1 = (R_xy + R_yx) / s_1
    qz_1 = (R_xz + R_zx) / s_1

    q_1 = tf.concat([tf.expand_dims(qw_1, axis=-1), tf.expand_dims(qx_1, axis=-1), tf.expand_dims(qy_1, axis=-1), tf.expand_dims(qz_1, axis=-1)], axis=-1)

    s_2 = tf.sqrt(tf.maximum(1.0 + R_yy - R_xx - R_zz, 1e-6)) * 2  # // S=4 * qy
    qw_2 = (R_xz - R_zx) / s_2
    qx_2 = (R_xy + R_yx) / s_2
    qy_2 = 0.25 * s_2
    qz_2 = (R_yz + R_zy) / s_2

    q_2 = tf.concat([tf.expand_dims(qw_2, axis=-1), tf.expand_dims(qx_2, axis=-1), tf.expand_dims(qy_2, axis=-1), tf.expand_dims(qz_2, axis=-1)], axis=-1)

    s_3 = tf.sqrt(tf.maximum(1.0 + R_zz - R_xx - R_yy, 1e-6)) * 2  # S=4 * qz
    qw_3 = (R_yx - R_xy) / s_3
    qx_3 = (R_xz + R_zx) / s_3
    qy_3 = (R_yz + R_zy) / s_3
    qz_3 = 0.25 * s_3

    q_3 = tf.concat([tf.expand_dims(qw_3, axis=-1), tf.expand_dims(qx_3, axis=-1), tf.expand_dims(qy_3, axis=-1), tf.expand_dims(qz_3, axis=-1)], axis=-1)

    s0_weight = tf.where(tf.greater(tr, 0.), tf.ones([batch_size]), tf.zeros([batch_size]))
    s1_weight = tf.where(tf.logical_and(tf.less(s0_weight, 1e-6), tf.logical_and(tf.greater(R_xx, R_yy), tf.greater(R_xx, R_zz))), tf.ones([batch_size]), tf.zeros([batch_size]))
    s2_weight = tf.where(tf.logical_and(tf.logical_and(tf.less(s0_weight, 1e-6), tf.less(s1_weight, 1e-6)), tf.greater(R_yy, R_zz)), tf.ones([batch_size]), tf.zeros([batch_size]))
    s3_weight = tf.ones([batch_size]) - (s0_weight + s1_weight + s2_weight)
    #s3_weight = tf.where(tf.less(s0_weight + s1_weight + s2_weight, 1e-6), tf.ones([batch_size]), tf.zeros([batch_size]))
    #s3_weight = tf.where(tf.logical_and(tf.less(s1_weight, 1e-6), tf.logical_and(tf.less(s1_weight, 1e-6), tf.less(s2_weight, 1e-6))), tf.ones([batch_size]), tf.zeros([batch_size]))

    with tf.control_dependencies([tf.debugging.assert_equal(s0_weight + s1_weight + s2_weight + s3_weight, 1.)]):
        q = tf.expand_dims(s0_weight, axis=-1) * q_0 + tf.expand_dims(s1_weight, axis=-1) * q_1 + tf.expand_dims(s2_weight, axis=-1) * q_2 + tf.expand_dims(s3_weight, axis=-1) * q_3


    return q / tf.maximum(tf.sqrt(tf.reduce_sum(q ** 2, axis=-1, keepdims=True)), epsilon)

def so3_to_euler(g, epsilon=1.e-6):

    g_flat = tf.reshape(g, [-1])

    batch_size = tf.shape(g)[0]

    r_xx_idx = tf.range(batch_size) * 9
    R_xx = tf.gather(g_flat, r_xx_idx)

    r_yx_idx = tf.range(batch_size) * 9 + 3
    R_yx = tf.gather(g_flat, r_yx_idx)

    r_yy_idx = tf.range(batch_size) * 9 + 4
    R_yy = tf.gather(g_flat, r_yy_idx)

    r_yz_idx = tf.range(batch_size) * 9 + 5
    R_yz = tf.gather(g_flat, r_yz_idx)

    r_zx_idx = tf.range(batch_size) * 9 + 6
    R_zx = tf.gather(g_flat, r_zx_idx)

    r_zy_idx = tf.range(batch_size) * 9 + 7
    R_zy = tf.gather(g_flat, r_zy_idx)

    r_zz_idx = tf.range(batch_size) * 9 + 8
    R_zz = tf.gather(g_flat, r_zz_idx)

    sy = R_xx * R_xx + R_yx * R_yx

    x_0 = tf.math.atan(R_zy / R_zz)
    y_0 = tf.math.atan(R_zx / (sy))
    z_0 = tf.math.atan(R_yx / R_xx)

    x_1 = tf.math.atan(-R_yz / R_yy)
    y_1 = tf.math.atan(R_zx / (sy))
    z_1 = tf.zeros_like((sy))

    euler_0 = tf.concat([tf.expand_dims(x_0, axis=-1), tf.expand_dims(y_0, axis=-1), tf.expand_dims(z_0, axis=-1)], axis=-1)
    euler_1 = tf.concat([tf.expand_dims(x_1, axis=-1), tf.expand_dims(y_1, axis=-1), tf.expand_dims(z_1, axis=-1)], axis=-1)

    s0_weight = tf.where(tf.less(sy, epsilon), tf.ones_like(sy), tf.zeros_like(sy))
    s1_weight = 1. - s0_weight

    return tf.expand_dims(s0_weight, axis=-1) * euler_0 + tf.expand_dims(s1_weight, axis=-1) * euler_1

def se3_to_components(g, out='matrix'):

    R, t = decompose(g)
    if out == 'quat':
        return so3_to_unit_quaternion(R), t
    elif out == 'euler':
        return so3_to_euler(R), t

    return R, t

def quaternion_to_so3(q):
    """
    :param q: [-1, 4] w, x, y, z
    :return:

    Following Gordon Pall, On The Rational Automorphs of x_1^2 + x_2^2 + x_3^2, Annals of Mathematics, volume 41, 1940
    """

    batch_size = tf.shape(q)[0]

    w, x, y, z = tf.split(q, 4, axis=-1)

    vals = [1. - 2. * y ** 2 - 2. * z ** 2, 2. * x * y - 2. * z * w, 2. * x * z + 2. * y * w,
            2. * x * y + 2. * z * w, 1. - 2. * x ** 2 - 2. * z ** 2, 2. * y * z - 2. * x * w,
            2. * x * z - 2. * y * w, 2. * y * z + 2. * x * w, 1. - 2. * x ** 2 - 2 * y ** 2]

    vals = tf.reshape(vals, [-1, batch_size])
    vals = tf.transpose(vals, perm=[1, 0])


    return tf.reshape(vals, [batch_size, 3, 3])

def avg_quaternions(q, w):
    """

    :param q: [-1, P, 4] w, x, y, z
    :param w: [-1, P] weight vector
    :return:

    Let
    Q = [a_1 * q_1 a_2 * q_2...a_n * q_n]

    Where a_i are the weight of the ith quaternion, and q_i are
    the ith quaternion being averaged, as a column vector. Q is therefore a 4xN matrix.

    The normalized eigenvector corresponding to the largest eigenvalue of
    Q * Q ^ T is the weighted average. Since Q * Q ^ T is self - adjoint and at
    least positive semi - definite, fast and robust methods
    of solving that eigenproblem are available. Computing the
    matrix - matrix product is the only step that
    grows  with the number of elements being averaged.
    """

    Q_t = q * tf.expand_dims(w, axis=-1)

    return eig.power_iteration(tf.matmul(tf.transpose(Q_t, perm=[0, 2, 1]), Q_t))

def se3_from_corrs(A, B, Weights, homogenous=True):
    """
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    :param A: [-1, num_points, dims]
    :param B: [-1, num_points, dims]
    :param Weights: [-1, num_points, 1]
    :return:
    """

    nb_segments = tf.shape(Weights)[-2]

    weight_sum = tf.reduce_sum(Weights, axis=-2, keepdims=True)

    mean_from = tf.reduce_sum(A * Weights, axis=-2, keepdims=True) / weight_sum
    mean_to = tf.reduce_sum(B * Weights, axis=-2, keepdims=True) / weight_sum

    mean_from = tf.reshape(mean_from, [-1, 1, 3])
    mean_to = tf.reshape(mean_to, [-1, 1, 3])

    # Get centroids of both in- and output and calculate deviation to centroids
    batch_size = tf.shape(A)[0]

    dev_from = A - mean_from
    dev_to = B - mean_to

    #weights_matrix = tf.eye(nb_segments, dtype=tf.float32, batch_shape=[batch_size])
    #weights_matrix *= Weights


    weights_matrix = tf.linalg.diag(tf.reshape(Weights, [batch_size, nb_segments]))

    H = tf.matmul(tf.transpose(dev_from, perm=[0, 2, 1]), weights_matrix)
    H = tf.matmul(H, dev_to)

    # Use SVD to calculate rotation and translation

    #FIXME: related to https://github.com/tensorflow/tensorflow/issues/17352 SVD not to run on GPU!!!!
    with tf.device('cpu:0'):
        _, u, v = tf.linalg.svd(H)

    #_, u, v = tf.linalg.svd(H, name='SVD')

    #_, u, v = tf.linalg.svd(H, full_matrices=False)

    R = tf.matmul(v, tf.transpose(u, perm=[0, 2, 1]))

    t = - tf.matmul(R, tf.transpose(mean_from, perm=[0, 2, 1])) + tf.transpose(mean_to, perm=[0, 2, 1])
    if homogenous:

        T = compose(R, t)

        return T

    return R, t

def acos(x):

    #acos_assert = tf.debugging.Assert(tf.logical_or(tf.greater(x, 1.0), tf.less(x, -1.)), x)
    #with tf.control_dependencies([acos_assert]):
    x = tf.clip_by_value(x, -1., 1.)
    return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966

def angle_between_rotation_matrices(P, Q):

    R = tf.matmul(P, tf.linalg.matrix_transpose(Q))

    return acos((tf.linalg.trace(R) - 1.) / 2.)

import transform as transform_util
import numpy as np

if __name__ == '__main__':

    epsilon = 1e-4

    batches = 16


    print ('Test so3 to quaternion')
    mat = tf.placeholder(shape=[None, 3, 3], dtype=tf.float32)

    q = so3_to_unit_quaternion(mat)

    np_mats = []
    np_q = []
    for i in range(batches):
        np_mat = transform_util.get_random_rotation(2. * np.pi)
        np_mats.append(np.expand_dims(np_mat[:3, :3], axis=0))
        np_q.append(transform_util.so3_to_quaternion_v2(np_mat))

    np_mats = np.concatenate(np_mats)

    with tf.Session() as sess:

        feed_dict = {mat: np_mats}
        tf_q = sess.run(q, feed_dict)

    print (np_q)
    print (tf_q)

    error = np.mean(np.sum((np_q - tf_q) ** 2, axis=-1))

    print (error)
    assert(error < epsilon)


    print ('Test quaternion to so3')
    q_tf = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    mat = quaternion_to_so3(q_tf)

    np_mats = []
    np_q = []

    for i in range(batches):
        q = (np.random.ranf(4) - .5) * 2.
        q /= np.sqrt(np.sum(q ** 2, axis=-1))
        np_mats.append(transform_util.quaternion_matrix(q))
        np_q.append(q)

    with tf.Session() as sess:

        feed_dict = {q_tf: np_q}
        tf_mat = sess.run(mat, feed_dict)




    error = np.matmul(tf_mat, np.linalg.inv(np_mats)) - np.expand_dims(np.eye(3), axis=0)
    error = np.mean(np.sum((np.reshape(error, (-1, 9))) ** 2, axis=-1))

    #error = np.mean(np.sum((np.reshape(np_mats - tf_mat, (-1, 9))) ** 2, axis=-1))

    print (error)
    assert(error < epsilon)





    print ('test transform_matrix_around_arbitrary_point')

    tranlation = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    mat = tf.placeholder(shape=[None, 4, 4], dtype=tf.float32)
    trans_mat = transform_matrix_around_arbitrary_point(mat, tranlation)


    np_t_mats = np.tile(np.expand_dims(np.eye(4), axis=0), (4, 1, 1))

    np_t_mats_neg = np.tile(np.expand_dims(np.eye(4), axis=0), (4, 1, 1))

    np_mats = []
    trans = []
    for i in range(4):
        np_mats.append(np.expand_dims(transform_util.get_random_transform(20., 2. * np.pi), axis=0))

        t = np.expand_dims((np.random.ranf(3) - .5) * 2., axis=0)
        np_t_mats[i, :3, 3] = t[0]
        np_t_mats_neg [i, :3, 3] = -t[0]

        trans.append(t)



    trans = np.concatenate(trans)
    np_mats = np.concatenate(np_mats)

    gt = np.matmul(np.matmul(np_t_mats, np_mats), np_t_mats_neg)


    with tf.Session() as sess:

        feed_dict = {tranlation: trans, mat: np_mats}
        tf_mats = sess.run([trans_mat], feed_dict)


    error = np.mean(np.sum(np.abs(np.reshape(gt - tf_mats, (-1, 16))), axis=-1))
    print (error)
    assert(error < epsilon)








    print ('test add_t')
    tranlation = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    mat = tf.placeholder(shape=[None, 4, 4], dtype=tf.float32)

    trans_mat = add_t(mat, tranlation)

    with tf.Session() as sess:


        np_t_mats = np.tile(np.expand_dims(np.eye(4), axis=0), (4, 1, 1))
        np_mats = []
        trans = []
        for i in range(4):
            np_mats.append(np.expand_dims(transform_util.get_random_transform(20., 2. * np.pi), axis=0))

            t = np.expand_dims((np.random.ranf(3) - .5) * 2., axis=0)
            np_t_mats[i, :3, 3] = t[0]

            trans.append(t)

        trans = np.concatenate(trans)
        np_mats = np.concatenate(np_mats)

        feed_dict = {tranlation: trans, mat: np_mats}
        tf_mats = sess.run([trans_mat], feed_dict)
        np_mats = np.matmul(np_mats, np_t_mats)

        error = np.mean(np.sum(np.abs(np.reshape(np_mats - tf_mats, (-1, 16))), axis=-1))
        print (error)
        assert(error < epsilon)



    print ('test inverse')
    trans_mat = tf.placeholder(shape=[None, 4, 4], dtype=tf.float32)

    trans_mat_inv = inverse(trans_mat)
    trans_mat_inv_gt = tf.linalg.inv(trans_mat)

    inv_diff = tf.reduce_sum(tf.abs(tf.reshape(trans_mat_inv - trans_mat_inv_gt, [-1, 16])), axis=-1)
    inv_diff = tf.reduce_mean(inv_diff)

    with tf.Session() as sess:

        mats = []
        for i in range(20):
            mats.append(np.expand_dims(transform_util.get_random_transform(20., 2. * np.pi), axis=0))
        mats = np.concatenate(mats)

        feed_dict = {trans_mat: mats}
        error = sess.run(inv_diff, feed_dict)
        print (error)
        assert (error < epsilon)

    print ('Test completed')