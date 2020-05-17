# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf
from tensorflow.python.keras.metrics import MAPE
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import (
    ModelCheckpoint,
    Callback,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau)

from data_utils import roll_df
from data import init_data_grade, grade_gene, gene_wrapper
from model import buildModel_vgg16


parser = argparse.ArgumentParser()
parser.add_argument("--train_name", type=str, help="Training/Experiment name")
parser.add_argument("--img_path", type=str, help="Path to train/test images")
parser.add_argument("--anno_file", type=str, help="Name of xlsx file containing annotations")
parser.add_argument("--patch_size", type=int, help="Height/Width Image Crop Size", default=320)
parser.add_argument("--batch_size", type=str, help="Training batch size", default=32)
args = parser.parse_args()

training_name = args.train_name
PATCH_SIZE = args.patch_size
BATCH_SIZE = args.batch_size

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

        
def train_():
    trn_data, val_data, tst_data = init_data_grade(args.anno_file) 
    for fold in range(3):
        print('-'*50)
        print('...Initializing classifier network...')
        print('-'*50)    
    
        model = buildModel_vgg16(
          input_dim=(PATCH_SIZE,PATCH_SIZE,3), 
          weights_path='imagenet')
        if fold == 0:
            model.summary()
        model.compile(
            optimizer=Adam(lr=1e-5), 
            loss={
                'BE': 'categorical_crossentropy',
                'ICM': 'binary_crossentropy',
                'TE': 'binary_crossentropy'},
            loss_weights=[1, 1, 1],
            metrics={
                'BE': 'categorical_accuracy',
                'ICM': 'categorical_accuracy',
                'TE': 'categorical_accuracy'})
        
        model_checkpoint = ModelCheckpoint(
            training_name + 'fold{}.hdf5'.format(fold), 
            monitor='val_loss', 
            save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=30)
        csv_logger = CSVLogger(training_name + 'logFold{}.csv'.format(fold))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=15, min_lr=1e-7)
        
        print('-'*50)
        print('...Gathering data...')
        print('-'*50)

        train_gene = grade_gene(
            data_df=trn_data, 
            img_path=args.img_path,
            batch_size=BATCH_SIZE, 
            target_size=(PATCH_SIZE, PATCH_SIZE),
            train=True)

        val_gene = grade_gene(
            data_df=val_data, 
            img_path=args.img_path,
            batch_size=BATCH_SIZE, 
            target_size=(PATCH_SIZE, PATCH_SIZE),
            train=False)
        
        print('-'*50)
        print('...Fitting model...')
        print('-'*50)

        model.fit_generator(
            generator=gene_wrapper(train_gene),
            steps_per_epoch=trn_data.shape[0] / BATCH_SIZE,
            epochs = 1000,
            callbacks = [
                model_checkpoint, 
                reduce_lr, 
                early_stop,
                csv_logger],
            validation_data=gene_wrapper(val_gene),
            validation_steps=val_data.shape[0] / BATCH_SIZE,
            verbose=1)

        trn_data, val_data, tst_data = roll_df(trn_data, val_data, tst_data, fold)


if __name__ == '__main__':
    train_()
