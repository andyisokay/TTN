import tensorflow as tf
import numpy as np
import ttn

# Enable tf 1.x compatible mode
tf.compat.v1.disable_eager_execution()

seq_len = 100
num_classes = 2


def weight_variable_zero_init(shape, name):
    w = tf.compat.v1.get_variable(
        name, shape=shape, initializer=tf.compat.v1.constant_initializer(0.0)
    )
    return w


def weight_variable(shape, name):
    w = tf.compat.v1.get_variable(
        name, shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling()
    )
    return w


def bias_variable(shape, name):
    b = tf.compat.v1.get_variable(
        name, shape=shape, initializer=tf.compat.v1.constant_initializer(0.1)
    )
    return b


def conv1d(x, W, stride):
    return tf.compat.v1.nn.conv1d(x, W, stride=stride, padding="SAME")


def mapping(sequence, batch_size):
    # Normalizing the sequence
    sequence1 = sequence - tf.compat.v1.tile(
        tf.compat.v1.reduce_mean(sequence, 1, keepdims=True), [1, seq_len]
    )
    sequence_norm = tf.compat.v1.sqrt(
        tf.compat.v1.reduce_sum(tf.compat.v1.square(sequence1), 1, keepdims=True)
    )
    sequence_norm_tile = tf.compat.v1.tile(sequence_norm, [1, seq_len])
    sequence1 = tf.compat.v1.div(sequence1, sequence_norm_tile)

    sequence1 = tf.compat.v1.reshape(sequence1, [batch_size, seq_len, 1])

    # TTN Structure
    with tf.compat.v1.variable_scope("ttn"):
        conv_size = 8
        num_channels = 1

        wg1 = weight_variable([conv_size, 1, num_channels], "wg1")
        bg1 = bias_variable([num_channels], "bg1")
        # a single 1D convolutional layer
        hg1 = tf.compat.v1.nn.tanh(conv1d(sequence1, wg1, 1) + bg1)

        print("hg1_size: ", hg1.shape)

        hg2 = tf.compat.v1.reshape(hg1, [batch_size, 100 * num_channels])

        print("hg2_size: ", hg2.shape)

        W_fc1g = weight_variable_zero_init([100 * num_channels, 99], "W_fc1g")
        b_fc1g = bias_variable([99], "b_fc1g")
        out1g = tf.compat.v1.nn.tanh(
            tf.compat.v1.nn.dropout(
                tf.compat.v1.matmul(hg2, W_fc1g) + b_fc1g, keep_prob=0.8
            )
        )

    # Constraint satisfaction layers
    temp = tf.compat.v1.sqrt(
        tf.compat.v1.reduce_sum(tf.compat.v1.square(out1g), 1, keepdims=True)
    )
    batch_temp = tf.compat.v1.tile(temp, [1, seq_len - 1])
    gamma_dot = tf.compat.v1.square(tf.compat.v1.div(out1g, batch_temp))
    gamma = tf.compat.v1.cumsum(gamma_dot, axis=1)
    zeros_vector = tf.compat.v1.zeros([batch_size, 1])
    gamma = tf.compat.v1.concat([zeros_vector, gamma], 1)
    gamma = gamma * (seq_len - 1)

    # Warping layer
    sequence_warped = ttn.warp(
        tf.compat.v1.reshape(sequence1, [batch_size, seq_len, 1]), gamma
    )
    sequence_warped = tf.compat.v1.reshape(sequence_warped, [batch_size, seq_len])

    tf.compat.v1.set_random_seed(1234)

    # Classifier
    with tf.compat.v1.variable_scope("classifier"):
        W_fc1 = weight_variable([100, num_classes], "W_fc1")
        b_fc1 = bias_variable([num_classes], "b_fc1")
        logits = tf.compat.v1.matmul(sequence_warped, W_fc1) + b_fc1

    return logits, sequence_warped, gamma, sequence1


def loss(logits, labels):
    cross_entropy = tf.compat.v1.reduce_mean(
        tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    return cross_entropy


def training(loss, learning_rate, var_list):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)
    return train_op
