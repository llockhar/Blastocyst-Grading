from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
import pandas as pd 
import numpy as np

IMG_PATH = 'Data/CroppedImages'

NUMBER = {4: 2, 3: 1, 2: 0}
LETTER = {'A': 2, 'B': 1, 'C': 0}

def init_data_grade(filename):
    data = pd.read_excel(
        filename, usecols=['Filename', 'BE', 'ICM', 'TE'])

    temp = list(zip(data['Filename'], data['BE'], data['ICM'], data['TE']))
    np.random.RandomState(421).shuffle(temp)
    data['Filename'], data['BE'], data['ICM'], data['TE'] = zip(*temp)

    # Assign labels to 0,1,2 for one-hot encoding
    data['BE']  = [NUMBER.get(item, item) for item in data['BE']]
    data['ICM'] = [LETTER.get(item, item) for item in data['ICM']]
    data['TE']  = [LETTER.get(item, item) for item in data['TE']]

    # Separate the data into classes
    data_0 = data[data['ICM'] == 0]
    data_1 = data[data['ICM'] == 1]
    data_2 = data[data['ICM'] == 2]

    # Partition the data into training, validation, and test sets 
    # representative of class distributions
    tst = 0.15
    data_1_len = data_1.shape[0]
    data_2_len = data_2.shape[0]
    tst_data = pd.concat([
        data_0[:2],
        data_1[:round(tst * data_1_len)],
        data_2[:round(tst * data_2_len)]])
    val_data = pd.concat([
        data_0[2:4],
        data_1[round(tst * data_1_len):round(2*tst * data_1_len)],
        data_2[round(tst * data_2_len):round(2*tst * data_2_len)]])
    trn_data = pd.concat([
        data_0[4:],
        data_1[round(2*tst * data_1_len):],
        data_2[round(2*tst * data_2_len):]])

    return trn_data, val_data, tst_data

def gene_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[to_categorical(batch_y[:,i], 3) for i in range(3)])

def grade_gene(data_df, batch_size, train, target_size):
    if train:
        datagen = ImageDataGenerator(
            rotation_range=360,
            width_shift_range=0.10,
            height_shift_range=0.10,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,)
    else:
        datagen = ImageDataGenerator()

    generator = datagen.flow_from_dataframe(
        dataframe=data_df,
        directory=IMG_PATH,
        x_col='Filename',
        y_col=['BE', 'ICM', 'TE'],
        target_size=target_size,
        color_mode='rgb',
        shuffle=train,
        seed=112,
        class_mode='raw',
        batch_size=batch_size)
    
    return generator
