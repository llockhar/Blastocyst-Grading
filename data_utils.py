import numpy as np 
import pandas as pd
import random
import os

def split_files(img_path):
    filenames = []
    for filename in os.listdir(img_path):
        filenames.append(filename)
    indices = list(np.arange(len(filenames)))
    random.RandomState(433).shuffle(indices)
    indices_train = indices[:round(0.70*len(indices))]  
    indices_val = indices[round(0.70*len(indices)):round(0.85*len(indices))]  
    indices_test = indices[round(0.85*len(indices)):]  

    return filenames, indices_train, indices_val, indices_test

def roll_df(trn_data, val_data, tst_data, fold):
    data = pd.concat([trn_data, val_data, tst_data])
    data_0 = np.asarray(data[data['ICM'] == 0])
    data_1 = np.asarray(data[data['ICM'] == 1])
    data_2 = np.asarray(data[data['ICM'] == 2])
    rolled_data_0 = np.roll(data_0, 2, axis=0)
    rolled_data_1 = np.roll(data_1, round(len(data_1)/3), axis=0)
    rolled_data_2 = np.roll(data_2, round(len(data_2)/3), axis=0)

    rolled_df_0 = pd.DataFrame({
        'Filename': rolled_data_0[:,0].astype(str), 
        'BE': rolled_data_0[:,1].astype(int),
        'ICM': rolled_data_0[:,2].astype(int),
        'TE': rolled_data_0[:,3].astype(int)})
    rolled_df_1 = pd.DataFrame({
        'Filename': rolled_data_1[:,0].astype(str), 
        'BE': rolled_data_1[:,1].astype(int),
        'ICM': rolled_data_1[:,2].astype(int),
        'TE': rolled_data_1[:,3].astype(int)})
    rolled_df_2 = pd.DataFrame({
        'Filename': rolled_data_2[:,0].astype(str), 
        'BE': rolled_data_2[:,1].astype(int),
        'ICM': rolled_data_2[:,2].astype(int),
        'TE': rolled_data_2[:,3].astype(int)})
    split = 0.15
    data_1_len = len(rolled_df_1)
    data_2_len = len(rolled_df_2)
    tst_data = pd.concat([
        rolled_df_0[:2],
        rolled_df_1[:round(split * data_1_len)],
        rolled_df_2[:round(split * data_2_len)]])
    val_data = pd.concat([
        rolled_df_0[2:4],
        rolled_df_1[round(split * data_1_len):round(2*split * data_1_len)],
        rolled_df_2[round(split * data_2_len):round(2*split * data_2_len)]])
    trn_data = pd.concat([
        rolled_df_0[4:],
        rolled_df_1[round(2*split * data_1_len):],
        rolled_df_2[round(2*split * data_2_len):]])

    return trn_data, val_data, tst_data

