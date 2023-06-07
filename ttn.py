import tensorflow as tf

# We need to do batch processing. gamma is a tensor of shape batch_size x sequence_length
# pose is the input of size batch_size x sequence_length x (num_joints*3), where 3 denotes the x,y,z coordinates
# gamma is of size batch_size x sequence_length
# output is the warped input, the same size as input

# Given gamma and the pose sequences, find the warped sequence, which is just linear interpolation done for each joint

# Enable tf 1.x compatible mode
tf.compat.v1.disable_eager_execution()


def warp(poses, gamma):
    batch_size = tf.compat.v1.shape(poses)[0]
    seq_len = tf.compat.v1.shape(poses)[1]
    pose_vec_len = tf.compat.v1.shape(poses)[2]

    zero = tf.compat.v1.zeros([], dtype="int32")
    max_gamma = seq_len - 1

    # do sampling
    gamma_0 = tf.compat.v1.cast(tf.compat.v1.floor(gamma), "int32")
    gamma_1 = gamma_0 + 1

    gamma_0 = tf.compat.v1.clip_by_value(gamma_0, zero, max_gamma)
    gamma_1 = tf.compat.v1.clip_by_value(gamma_1, zero, max_gamma)

    # Tile the gammas for each of the joint elements, so each gamma_0 and gamma_1 now becomes batch_size x sequence_length x (num_joints*3)
    gamma_tile = tf.compat.v1.tile(
        tf.compat.v1.expand_dims(gamma, 2), [1, 1, pose_vec_len]
    )
    gamma_0_tile = tf.compat.v1.tile(
        tf.compat.v1.expand_dims(gamma_0, 2), [1, 1, pose_vec_len]
    )
    gamma_1_tile = tf.compat.v1.tile(
        tf.compat.v1.expand_dims(gamma_1, 2), [1, 1, pose_vec_len]
    )

    # interpolation
    poses_flat = tf.compat.v1.reshape(poses, [batch_size * seq_len, pose_vec_len])
    gamma_0_flat = tf.compat.v1.reshape(gamma_0, [batch_size * seq_len])
    gamma_1_flat = tf.compat.v1.reshape(gamma_1, [batch_size * seq_len])

    # Need to add position_in_batch*seq_len to each element in gamma_0_flat and gamma_1_flat. This is required for tf.compat.v1.gather
    range_vec = tf.compat.v1.range(batch_size)
    range_vec_tile = tf.compat.v1.tile(
        tf.compat.v1.expand_dims(range_vec, 1), [1, seq_len]
    )
    range_vec_tile_vec = tf.compat.v1.reshape(range_vec_tile, [batch_size * seq_len])
    offset = range_vec_tile_vec * seq_len

    Ia_flat = tf.compat.v1.gather(poses_flat, gamma_0_flat + offset)
    Ib_flat = tf.compat.v1.gather(poses_flat, gamma_1_flat + offset)

    Ia = tf.compat.v1.reshape(Ia_flat, [batch_size, seq_len, pose_vec_len])
    Ib = tf.compat.v1.reshape(Ib_flat, [batch_size, seq_len, pose_vec_len])

    gamma_0_tile = tf.compat.v1.cast(gamma_0_tile, dtype="float32")
    gamma_1_tile = tf.compat.v1.cast(gamma_1_tile, dtype="float32")

    wa = 1 - (gamma_tile - gamma_0_tile)
    wb = 1 - wa

    output = tf.multiply(wa, Ia) + tf.multiply(wb, Ib)

    return output
