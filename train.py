from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import os
import sys
import pathlib
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
from settings import *

tf.logging.set_verbosity(tf.logging.INFO)

DATA_DIR = os.path.join(ROOT_DIR, TRAINING_IMAGES_DIR)
TEST_DIR = os.path.join(ROOT_DIR, TESTING_IMAGES_DIR)
OUTPUT_LABELS = os.path.join(ROOT_DIR, LABEL_FILENAME)
OUTPUT_GRAPH = os.path.join(ROOT_DIR, GRAPH_FILENAME)
GRAPH_DIR = os.path.join(ROOT_DIR, GRAPH_LOC)
assert(CHANNELS in [1,3])

def main(_):
	# Get the training and testing datasets and associated variables
    ds, num_images, label_names, label_to_index = getData()
    test_ds, test_num_images, test_label_names, test_label_to_index = getData('test')
	
	# Shuffle the datasets and prepare an iterator for training
    ds = ds.shuffle(buffer_size=num_images)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=num_images)
    iterator = ds.make_initializable_iterator()
    next_element = iterator.get_next()
    test_ds = test_ds.shuffle(buffer_size=test_num_images)
    test_ds = test_ds.batch(test_num_images)
    test_iterator = test_ds.make_one_shot_iterator().get_next()
	
	###
	#	The rest was taken from mnist_deep.py except where noted by initials CAA
	###
    # Create the model
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS]) # CAA: Modified parameters to be customizable

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x, len(label_names)) # CAA: Modified input arguments to include variable number of classes

	# CAA: Creating a final tensor to be used by the "test.py" script
    final_tensor = tf.nn.softmax(y_conv, name=FINAL_TENSOR_NAME)
	

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = GRAPH_DIR # CAA: Made graph locations customizable for debugging purposes
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS): # CAA: Made number of steps customizable
            sess.run(iterator.initializer) # CAA: Initializing training iterator each step
            batch = sess.run(next_element) # CAA: Grabbing data
            if i % (EPOCHS/10) == 0: # CAA: Made accuracy predictions set to 10 per total run at even intervals
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        test_batch = sess.run(test_iterator) # CAA: Grabbing test data
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})) # CAA: Modified data input argument

        ###
		# Pulled from retrain.py, modified to work with this script
		###
		# write out the trained graph and labels with the weights stored as constants
        print("writing trained graph and labels with weights")
        save_graph_to_file(sess, tf.get_default_graph(), OUTPUT_GRAPH) # CAA: Modified arguments to fit with this data
        with gfile.FastGFile(OUTPUT_LABELS, 'w') as f:
            f.write('\n'.join(label_names) + '\n')
        # end with

    return None

#######################################################################################################################
###
# Pulled from retrain.py
###
def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    # end with
    return
# end function

# CAA: Processing function adapted from https://www.tensorflow.org/tutorials/load_data/images
# Takes an image and modifies the data to fit desired parameters
def preprocess_image(image):
    tf_decoded = tf.image.decode_jpeg(image, channels=3)
    tf_bgr = tf_decoded[:,:,::-1]
    if CHANNELS == 1: # CAA: converts to grayscale if settings.py indicated that desire
        tf_bgr = tf.image.rgb_to_grayscale(tf_bgr)
    tf_resized = tf.image.resize_images(tf_bgr, [IMAGE_SIZE, IMAGE_SIZE]) # CAA: resizes based on desires in settings.py
    return tf.cast(tf_resized, dtype=tf.float32)

# CAA: Processing function adapted from https://www.tensorflow.org/tutorials/load_data/images
# Takes a path to an image file, loads modifies the image data to fit desired parameters
def load_and_preprocess_image(path):
    tf_img_string = tf.read_file(path)
    return preprocess_image(tf_img_string)

# CAA: Data retrieval function adapted from https://www.tensorflow.org/tutorials/load_data/images
# If 'test' is passed as an argument, returns testing dataset; otherwise will return training dataset
def getData(source = 'train'):
    if (source == 'test'): data_root = pathlib.Path(TEST_DIR)
    else: data_root = pathlib.Path(DATA_DIR)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]

    num_images = len(all_image_paths)

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_ds = path_ds.map(load_and_preprocess_image)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds, num_images, label_names, label_to_index

###
# Taken from mnist_deep.py and modified as indicated in comments tagged with initials CAA
###
def deepnn(x, nLabels):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), where
      IMAGE_SIZE, IMAGE_SIZE, CHANNELS is the shape of the image

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, nLabels), with values
      equal to the logits of classifying the digit into one of nLabels classes (the
      digits 0-(nLabels-1)). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - 
    # -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS]) # CAA: Customized image dimensions

    # First convolutional layer - maps one image to 32 feature maps per channel.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, CHANNELS, 32*CHANNELS]) # CAA: Customized number of channels
        b_conv1 = bias_variable([32*CHANNELS]) # CAA: Customized number of channels
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps per channel to 64 per channel.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32*CHANNELS, 64*CHANNELS]) # CAA: Customized number of channels
        b_conv2 = bias_variable([64*CHANNELS]) # CAA: Customized number of channels
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our IMAGE_SIZExIMAGE_SIZE image
    # is down to (IMAGE_SIZE/4)x(IMAGE_SIZE/4)x(64*CHANNELS) feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([int((IMAGE_SIZE/4) * (IMAGE_SIZE/4) * (64 * CHANNELS)), 1024])  # CAA: Customized image dimensions
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, int((IMAGE_SIZE/4) * (IMAGE_SIZE/4) * (64 * CHANNELS))]) # CAA: Customized image dimensions
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, nLabels]) # CAA: Customized number of labels
        b_fc2 = bias_variable([nLabels]) # CAA: Customized number of labels

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob

###
# Taken from mnist_deep.py
###
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

###
# Taken from mnist_deep.py
###
def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

###
# Taken from mnist_deep.py
###
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

###
# Taken from mnist_deep.py
###
def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)
