import tensorflow as tf
import numpy as np
import network_ttn_synthetic as network
import os.path
import scipy.io

# Enable tf 1.x compatible mode
tf.compat.v1.disable_eager_execution()

batch_size = 1
seq_len = 100
num_classes = 2

test_data = scipy.io.loadmat("synthetic_data_test_2_gaussians.mat")["test_data"]
test_label = scipy.io.loadmat("synthetic_data_test_2_gaussians.mat")["test_label"]


def placeholder_inputs(batch_size):
    x_placeholder = tf.compat.v1.placeholder(
        tf.compat.v1.float32, shape=(batch_size, seq_len)
    )
    y_placeholder = tf.compat.v1.placeholder(
        tf.compat.v1.float32, shape=(batch_size, num_classes)
    )

    return x_placeholder, y_placeholder


with tf.compat.v1.Graph().as_default():
    checkpointPath = "./2_gaussians_github_ttn"

    x_placeholder, y_placeholder = placeholder_inputs(batch_size)
    output, sequence_unwarped, gamma, sequence1 = network.mapping(
        x_placeholder, batch_size
    )
    correct_prediction = tf.compat.v1.equal(
        tf.compat.v1.argmax(output, 1), tf.compat.v1.argmax(y_placeholder, 1)
    )

    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    saver.restore(sess, checkpointPath)

    correct = 0
    generated_gamma = np.zeros((test_data.shape[0], 100))
    ttn_output = np.zeros((test_data.shape[0], 100))
    sequence_normalized = np.zeros((test_data.shape[0], 100))

    for testExample in range(test_data.shape[0]):
        output_val, ttn_output_val, gamma_val, sequence1_val, correct_val = sess.run(
            [output, sequence_unwarped, gamma, sequence1, correct_prediction],
            feed_dict={
                x_placeholder: np.reshape(test_data[testExample, :], [1, seq_len]),
                y_placeholder: np.reshape(test_label[testExample, :], [1, num_classes]),
            },
        )
        correct = correct + int(correct_val[0] == True)

        generated_gamma[testExample, :] = gamma_val
        ttn_output[testExample, :] = ttn_output_val
        sequence_normalized[testExample, :] = np.squeeze(sequence1_val)

accuracy = correct / float(test_data.shape[0])

scipy.io.savemat("./ttn_output_github.mat", mdict={"ttn_output": ttn_output})
scipy.io.savemat(
    "./generated_gamma_github.mat", mdict={"generated_gamma": generated_gamma}
)
scipy.io.savemat(
    "./sequence_normalized_github.mat",
    mdict={"sequence_normalized": sequence_normalized},
)

print("Accuracy:")
print(accuracy)
