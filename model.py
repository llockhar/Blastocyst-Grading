# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.utils import get_file
from keras.regularizers import l2
import tensorflow as tf

from model_blocks import (
  conv_bn_relu, _conv_bn_relu_x2, 
  grading_block_conv, conv_dil_block,
  conv_dil_block_bn)


WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    

def vgg16_encoder(input_dim, weights_path=None):
    img_input = Input(shape=input_dim)
    # Block 1
    x = Conv2D(64,(3, 3),activation='relu',padding='same',name='block1_conv1')(img_input)
    x = Conv2D(64,(3, 3),activation='relu',padding='same',name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128,(3, 3),activation='relu',padding='same',name='block2_conv1')(x)
    x = Conv2D(128,(3, 3),activation='relu',padding='same',name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256,(3, 3),activation='relu',padding='same',name='block3_conv1')(x)
    x = Conv2D(256,(3, 3),activation='relu',padding='same',name='block3_conv2')(x)
    x = Conv2D(256,(3, 3),activation='relu',padding='same',name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block4_conv1')(x)
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block4_conv2')(x)
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block5_conv1')(x)
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block5_conv2')(x)
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(img_input, x, name='vgg16')
    if weights_path is not None:
        if weights_path == 'imagenet':
            weights = get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        else:
            weights = weights_path
        model.load_weights(weights, by_name=True)

    return model


def vgg16_top(input_img, app):
    x = input_img
    for layer in app.layers[:-3]: 
        layer.trainable = False
    for layer in app.layers:
      x = layer(x)
    
    out1 = grading_block('BE')(x)
    out2 = grading_block('ICM')(x)
    out3 = grading_block('TE')(x)
    
    return out1, out2, out3

def grading_block(grade):
    def l(input_layer):
        avg = GlobalAveragePooling2D()(input_layer)
        dense = Dense(units=32, activation='relu', name='dense_' + grade)(avg)
        drop = Dropout(rate=0.5, name='drop_' + grade)(dense)
        out = Dense(3, activation='softmax', name=grade)(drop)

        return out
    return l


def buildModel_vgg16 (input_dim, weights_path=None):
    input_img = Input(shape=(input_dim))

    vgg16_bottom = vgg16_encoder(input_dim, weights_path)
    out1, out2, out3 = vgg16_top(input_img, vgg16_bottom)

    model = Model(inputs=input_img, outputs=[out1, out2, out3])
    
    return model
