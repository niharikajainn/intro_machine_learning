import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

image_size = 28
image_as_vector = image_size * image_size  # 784
num_classes = 10  # 0-9

x = tf.placeholder(tf.float32, shape=[None, image_as_vector])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
y_true_class = tf.placeholder(tf.int64, shape=[None])

def network_model(x):
    weights = {
                'W_layer1': tf.Variable(tf.random_normal([image_as_vector, 600])),
                'W_layer2': tf.Variable(tf.random_normal([600, 500])),
                'W_layer3': tf.Variable(tf.random_normal([500, 400])),
                'logits': tf.Variable(tf.random_normal([400, num_classes]))
              }
    biases = {
                'b_layer1': tf.Variable(tf.random_normal([600])),
                'b_layer2': tf.Variable(tf.random_normal([500])),
                'b_layer3': tf.Variable(tf.random_normal([400])),
                'logits': tf.Variable(tf.random_normal([num_classes]))
             }

    layer1 = tf.matmul(x, weights['W_layer1']) + biases['b_layer1']
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.matmul(layer1, weights['W_layer2']) + biases['b_layer2']
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.matmul(layer2, weights['W_layer3']) + biases['b_layer3']
    layer3 = tf.nn.relu(layer3)
    logits = tf.matmul(layer3, weights['logits']) + biases['logits']

    y_predicted = tf.nn.softmax(logits)
    return logits, y_predicted


def train_and_test_network():
    logits, y_predicted = network_model(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    y_predicted_class = tf.argmax(y_predicted, axis=1)
    correctness = tf.equal(y_predicted_class, y_true_class) # returns vector of type bool
    accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 100

    for epoch in range(20):

        loss_over_all_data = 0
        for i in range(len(mnist.train.labels)/batch_size):
            x_next_batch, y_next_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run(
                [optimizer, loss],
                feed_dict={x: x_next_batch, y_true: y_next_batch})
            loss_over_all_data += loss_batch
        print "\nEpoch " + str(epoch + 1)
        print "Loss: " + str(round(loss_over_all_data,2))

    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_labels_class = np.argmax(test_labels, axis=1)

    accuracy_test, correctness_test, y_predicted_class_test = sess.run(
        [accuracy, correctness, y_predicted_class],
        feed_dict={x: test_images, y_true_class: test_labels_class})

    print "\nTotal Accuracy: " + str(round(accuracy_test * 100,2)) + "%"

    return correctness_test, y_predicted_class_test, test_labels_class


def plot_sample_input():
    rand = random.randint(0,len(mnist.train.images))
    image = np.reshape(mnist.train.images[rand], (image_size, image_size))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap="gray_r")
    plt.xlabel("Example image. True value: [" + str(np.argmax(mnist.train.labels[rand], axis=0)) + "]")
    plt.show()

def plot_sample_error(correctness_test, predicted_labels, test_labels_class):
    indices = np.where(correctness_test == False)[0]
    rand = random.randint(0,len(indices))
    image = np.reshape(mnist.test.images[indices[rand]], (image_size, image_size))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap="gray_r")
    plt.xlabel("Predicted value: " + str(predicted_labels[indices[rand]]) + ". True value: " + str(test_labels_class[indices[rand]]))
    plt.show()


plot_sample_input()
correctness_test, y_predicted_class_test, test_labels_class = train_and_test_network()
plot_sample_error(correctness_test, y_predicted_class_test, test_labels_class)
