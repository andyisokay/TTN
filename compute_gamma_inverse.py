# This function inverts a warping function batch-wise
# Input: gamma is a B x N array where B is the batch size and N is the sequence length

import tensorflow as tf
import numpy as np

# Enable tf 1.x compatible mode
tf.compat.v1.disable_eager_execution()


def invert_gamma(gamma, batch_size, seq_len):
    """
    a = np.array([[0.0,0.8,2.0],[0.0,1.2,2.0]])

    gamma = tf.compat.v1.constant(a)

    batch_size = gamma.get_shape()[0] #tf.compat.v1.shape(gamma)[0]
    seq_len = gamma.get_shape()[1] #tf.compat.v1.shape(gamma)[1]

    batch_size = 2
    seq_len = 3
    """

    false_vector = tf.compat.v1.constant(
        dtype=tf.compat.v1.bool, shape=[batch_size * (seq_len - 2), 1], value=False
    )

    input_indices = tf.compat.v1.reshape(
        tf.compat.v1.range(1, seq_len - 1), [seq_len - 2, 1]
    )
    input_indices_tile = tf.compat.v1.tile(input_indices, [batch_size, 1])
    input_indices_tile_col = tf.compat.v1.tile(input_indices_tile, [1, seq_len])

    gamma = tf.compat.v1.reshape(gamma, [batch_size, seq_len, 1])
    gamma_tile = tf.compat.v1.tile(gamma, [1, 1, seq_len - 2])
    gamma_transpose = tf.compat.v1.transpose(gamma_tile, [0, 2, 1])
    gamma_reshape = tf.compat.v1.reshape(
        gamma_transpose, [batch_size * (seq_len - 2), seq_len]
    )

    tempa = tf.compat.v1.less_equal(
        gamma_reshape, tf.compat.v1.cast(input_indices_tile_col, tf.compat.v1.float64)
    )
    tempb = tf.compat.v1.greater(
        gamma_reshape, tf.compat.v1.cast(input_indices_tile_col, tf.compat.v1.float64)
    )

    not_tempa_ext = tf.compat.v1.logical_not(
        tf.compat.v1.slice(
            tf.compat.v1.concat(1, (tempa, false_vector)),
            [0, 1],
            [batch_size * (seq_len - 2), seq_len],
        )
    )
    not_tempb_ext = tf.compat.v1.logical_not(
        tf.compat.v1.slice(
            tf.compat.v1.concat(1, (false_vector, tempb)),
            [0, 0],
            [batch_size * (seq_len - 2), seq_len],
        )
    )

    tempa_and = tf.compat.v1.logical_and(tempa, not_tempa_ext)
    tempb_and = tf.compat.v1.logical_and(tempb, not_tempb_ext)

    temp1 = tf.compat.v1.where(tempa_and)
    temp2 = tf.compat.v1.where(tempb_and)

    index1 = tf.compat.v1.slice(
        temp1, [0, 1], [batch_size * (seq_len - 2), 1]
    )  # temp1[:,0]
    index2 = tf.compat.v1.slice(
        temp2, [0, 1], [batch_size * (seq_len - 2), 1]
    )  # temp2[:,0]

    # Now do interpolation
    gamma_reshape_flat = tf.compat.v1.reshape(
        gamma_reshape, [batch_size * (seq_len - 2) * seq_len]
    )
    index1_flat = tf.compat.v1.reshape(index1, [batch_size * (seq_len - 2)])
    index2_flat = tf.compat.v1.reshape(index2, [batch_size * (seq_len - 2)])

    # The offset vector for tf.compat.v1.gather
    range_vec = tf.compat.v1.range(batch_size)
    range_vec_tile = tf.compat.v1.tile(
        tf.compat.v1.expand_dims(range_vec, 1), [1, seq_len - 2]
    )  # CHECK
    range_vec_tile_vec = tf.compat.v1.reshape(
        range_vec_tile, [batch_size * (seq_len - 2)]
    )
    offset = tf.compat.v1.cast(
        range_vec_tile_vec * (seq_len), tf.compat.v1.int64
    )  # CHECK

    gamma_index1 = tf.compat.v1.gather(gamma_reshape_flat, index1_flat + offset)
    gamma_index2 = tf.compat.v1.gather(gamma_reshape_flat, index2_flat + offset)

    input_indices_tile_flat = tf.compat.v1.reshape(
        input_indices_tile, [batch_size * (seq_len - 2)]
    )

    temp_alpha = tf.compat.v1.div(
        tf.compat.v1.cast(index2_flat, tf.compat.v1.float64)
        - tf.compat.v1.cast(index1_flat, tf.compat.v1.float64),
        gamma_index2 - gamma_index1,
    )
    index = tf.compat.v1.squeeze(
        tf.compat.v1.cast(index1, tf.compat.v1.float64)
    ) + tf.compat.v1.mul(
        temp_alpha,
        (
            tf.compat.v1.cast(input_indices_tile_flat, tf.compat.v1.float64)
            - gamma_index1
        ),
    )
    index_reshape = tf.compat.v1.reshape(index, [batch_size, seq_len - 2])

    gamma_inverse = tf.compat.v1.concat(
        1,
        (
            tf.compat.v1.zeros((batch_size, 1)),
            tf.compat.v1.cast(index_reshape, tf.compat.v1.float32),
            (seq_len - 1) * tf.compat.v1.ones((batch_size, 1)),
        ),
    )

    return gamma_inverse


# sess=tf.compat.v1.Session()
# chk = gamma_inverse.eval(session=sess)
# print chk
