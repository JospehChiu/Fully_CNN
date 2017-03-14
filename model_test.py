import tensorflow as tf
import sys
import time
import pickle
import scipy.misc
import numpy as np

from model import fcn_16s
from utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from utils.inference import adapt_network_for_any_size_input

slim = tf.contrib.slim

test_filename = 'valid.tfrecords'
number_of_classes = 21

test_queue = tf.train.string_input_producer([test_filename], num_epochs=1)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(test_queue)

image_batch = tf.expand_dims(image, axis=0)
annotation_batch = tf.expand_dims(annotation, axis=0)

fcn_16s = adapt_network_for_any_size_input(fcn_16s, 32)

pred, fcn_16s_variables_mapping = fcn_16s(image_batch_tensor=image_batch,
                                          num_classes=number_of_classes,
                                          is_training=False)

weights = tf.to_float( tf.not_equal(annotation_batch, 255) )

m_iou, update_op = slim.metrics.streaming_mean_iou(predictions=pred, labels=annotation_batch, num_classes=number_of_classes, weights=weights)

local_vars_init_op = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(local_vars_init_op)

    saver.restore(sess, "/home/jochiu/Fully_CNN/model_16s.ckpt")
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    count = 0
    tic = time.time()
    for i in range(1110):
        
        image_np, annotation_np, pred_np, tmp = sess.run([image, annotation, pred, update_op])
        upsampled_predictions = pred_np.squeeze()
        if np.sum(upsampled_predictions) != 0 and count <= 10:
            count += 1
            scipy.misc.imsave('visualize16/' + str(i) + '.png', image_np)
            with open('visualize16/' + str(i) + '.pickle', 'wb') as f:
                pickle.dump(upsampled_predictions, f)
    
    toc = time.time()
    coord.request_stop()
    coord.join(threads)
    res = sess.run(m_iou)
    
    print("Mean IOU: " + str(res))
    print("Average time: " + str((toc-tic)/1111))
