import tensorflow as tf
import numpy as np
from utils.upsampling import bilinear_upsample_weights

slim = tf.contrib.slim
vgg = tf.contrib.slim.python.slim.nets

def fcn_16s(image_batch, num_classes, is_training):
    # Get the filters for upsampling by factor 2 and 16
    upsample_by_2_weights = bilinear_upsample_weights(factor = 2, 
                                                      number_of_classes = num_classes)
    upsample_by_2_filter = tf.constant(upsample_by_2_weights)
    upsample_by_16_weights = bilinear_upsample_weights(factor = 16,
                                                       number_of_classes = num_classes)
    upsample_by_16_filter = tf.constant(upsample_by_16_weights)
    
    
    # Create a variable scope for our model
    with tf.variable_scope('fcn_16s') as fcn_16s_scope:
        # arg_scope defines the default functions for layer variables such as initailzers
        with slim.arg_scope(vgg.arg_scope()):
            # tensorflow slim vgg_16 signature: 
            # def vgg_16(inputs, num_classes=1000, is_training=True,
            #            dropout_keep_prob=0.5, spatial_squeeze=True, scope='vgg_16'):
            # Need to use 'same' padding for convolutional layers in vgg
            vgg_logits, vgg_endpoints = vgg.vgg_16(image_batch, 
                                                   num_classes = num_classes,
                                                   is_training = is_training,
                                                   spatial_squeeze = False)
            vgg_layer_shape = tf.shape(vgg_logits)
            
            
            # Calculate the size of the tensor upsampled by two times
            # vgg_layer_shape[0] is the batch size
            # vgg_layer_shape[1] is the height and vgg_layer_shape[2] is the width
            upsample_by_2_shape = tf.pack(vgg_layer_shape[0],
                                          vgg_layer_shape[1] * 2,
                                          vgg_layer_shape[2] * 2,
                                          vgg_layer_shape[3])
            # Perform upsampling using transpose convolution
            # conv2d_transpose input: 
            # tf.nn.conv2d_transpose(value, filter, output_shape, strides, 
            #                       padding='SAME', data_format='NHWC', name=None)
            upsample_by_2_logits = tf.nn.conv2d_transpose(vgg_logits, upsample_by_2_filter,
                                                          upsample_by_2_shape, [1, 2, 2, 1])       
            
            # Now we add the skip layer from pool4 layer of vgg
            pool4_features = vgg_endpoints['fcn_16s/vgg_16/pool4']
        
            # The pool4 output, according to paper, needs to go through a 
            # convolutional layer before being combined with the FCN32 logits
            pool4_logits = slim.conv2d(pool4_features, 
                                       num_classes, 
                                       [1, 1],
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       weights_initializer=tf.zeros_initializer(),
                                       scope='pool4_fc')
            vgg_pool4_combined_logits = upsample_by_2_logits + pool4_logits
            
            # Now upsample the combined logits by a factor of 16
            upsample_by_16_shape = tf.pack(vgg_pool4_combined_logits[0],
                                           vgg_pool4_combined_logits[1] * 16,
                                           vgg_pool4_combined_logits[2] * 16,
                                           vgg_pool4_combined_logits[3])
            upsample_by_16_logits = tf.nn.conv2d_transpose(vgg_pool4_combined_logits,
                                                           upsample_by_16_filter,
                                                           upsample_by_16_shape, 
                                                           [1, 16, 16, 1])
            
            # We need this mapping to load pretrained vgg model
            vgg_16_variables_mapping = {}
            fcn_16s_variables = slim.get_variables(fcn_16s_scope)
            
            for variable in fcn_16s_variables:
                
                # We only need FCN-32s variables to resture from checkpoint
                # Variables of FCN-16s should be initialized
                if 'pool4_fc' in variable.name:
                    continue

                # Here we remove the part of a name of the variable
                # that is responsible for the current variable scope
                original_vgg_16_checkpoint_string = variable.name[len(fcn_16s_scope.original_name_scope):-2]
                vgg_16_variables_mapping[original_vgg_16_checkpoint_string] = variable


    return upsample_by_16_logits, vgg_16_variables_mapping 

