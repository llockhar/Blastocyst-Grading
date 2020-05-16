import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
from os.path import join, exists

from keras_preprocessing.image.utils import array_to_img, img_to_array, load_img
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb

from data import grade_gene, gene_wrapper, init_data_grade
from model import buildModel_vgg16
from data_utils import roll_df

from sklearn.metrics import (
    balanced_accuracy_score, precision_score,
    recall_score, jaccard_score,
    confusion_matrix, classification_report)

REBMUN = {2:  4,  1:  3,  0:  2}
RETTEL = {2: 'A', 1: 'B', 0: 'C'}

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

OUT_PATH = training_name[-8:] + '/Predictions/'
CAMAP_PATH = training_name[-8:] + '/CAMAPS'

if not exists(OUT_PATH):
    os.makedirs(OUT_PATH)
if not exists(CAMAP_PATH):
    os.mkdir(CAMAP_PATH)


def get_iterates(i, class_idx, model, last_conv_layer):

    class_output = model.output[i][:,class_idx]
    grads = K.gradients(class_output, last_conv_layer.get_output_at(1))[0] 
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function(
      [model.input], 
      [pooled_grads, last_conv_layer.get_output_at(1)[0]])

    return iterate

def visualize_camap(model, fold, test_imgs):

    last_conv_layer = model.get_layer('block5_conv3')
    iterateBE0 = get_iterates(0, 0, model, last_conv_layer)
    iterateBE1 = get_iterates(0, 1, model, last_conv_layer)
    iterateBE2 = get_iterates(0, 2, model, last_conv_layer)
    iterateICM0 = get_iterates(1, 0, model, last_conv_layer)
    iterateICM1 = get_iterates(1, 1, model, last_conv_layer)
    iterateICM2 = get_iterates(1, 2, model, last_conv_layer)
    iterateTE0 = get_iterates(2, 0, model, last_conv_layer)
    iterateTE1 = get_iterates(2, 1, model, last_conv_layer)
    iterateTE2 = get_iterates(2, 2, model, last_conv_layer)

    CAM_DICT = {
      "BE0": iterateBE0,
      "BE1": iterateBE1,
      "BE2": iterateBE2,
      "ICM0": iterateICM0,
      "ICM1": iterateICM1,
      "ICM2": iterateICM2,
      "TE0": iterateTE0,
      "TE1": iterateTE1,
      "TE2": iterateTE2}

    for img_name in test_imgs:
        print(img_name)
        img_arr = load_img(
            join(args.img_path, img_name), 
            color_mode='rgb',
            target_size=(PATCH_SIZE, PATCH_SIZE)
        )
        img_arr = img_to_array(img_arr)
        x = np.expand_dims(img_arr, axis=0)
        # x = preprocess_input(x)

        img = imread(join(args.img_path, img_name))
        img = np.uint8(255 * resize(img, (img_arr.shape[1], img_arr.shape[0])))

        preds = model.predict(x)
        
        for i, grade in enumerate(['BE', 'ICM', 'TE']):
            iterate = CAM_DICT["{}{}".format(grade,np.argmax(preds[i]))]
            pooled_grads_value, conv_layer_output_value = iterate([x])

            pooled_grads_value_resized = pooled_grads_value * np.ones(conv_layer_output_value.shape)
            temp = np.multiply(conv_layer_output_value, pooled_grads_value_resized)

            heatmap = np.mean(temp, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + K.epsilon())
            heatmap = resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            
            plt.figure()
            plt.imshow(heatmap, cmap=plt.get_cmap('jet'), alpha=0.4)
            plt.imshow(gray2rgb(img), alpha=0.6)
            cur_ax = plt.gca()
            cur_ax.axes.get_xaxis().set_visible(False)
            cur_ax.axes.get_yaxis().set_visible(False)
            plt.savefig(join(CAMAP_PATH, img_name[:-4] + 'fold{}'.format(fold) + grade + img_name[-4:]),
                        bbox_inches='tight', pad_inches=0)
            plt.close()

def test_():

    trn_data, val_data, tst_data = init_data_grade(args.anno_file) 
    total_preds = None
    total_trues = None
    total_fnames = []

    for fold in range(3):
        # print(test_indices)
        
        print('-'*50)
        print('...Initializing classifier network...')
        print('-'*50)    
    
        model = buildModel_vgg16(
          input_dim=(PATCH_SIZE,PATCH_SIZE,3), 
          weights_path=None)
        model.load_weights(
          training_name + 'fold{}.hdf5'.format(fold), 
          by_name=True)
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
                'ICM': 'acc',
                'TE': 'acc'}
        )
        # model.summary()
        print('-'*50)
        print('...Evaluating model...')
        print('-'*50)

        test_gene = grade_gene(
            data_df=tst_data, 
            img_path=args.img_path,
            batch_size=BATCH_SIZE, 
            target_size=(PATCH_SIZE, PATCH_SIZE),
            train=False)

        print('Test dataframe: ', tst_data.head(), tst_data.shape[0])

        visualize_camap(model, fold, tst_data['Filename'].values.tolist())

        # Prediction/Evaluation
        test_gene.reset()
        preds = model.predict_generator(
            gene_wrapper(test_gene), 
            steps=tst_data.shape[0] / BATCH_SIZE)
        y_pred_labels = np.concatenate(
            # (np.array(preds)[0,:,:], np.array(preds)[1,:,:], np.array(preds)[2,:,:]),
            (np.array(preds[0]), np.array(preds[1]), np.array(preds[2])),
            axis=1)
        y_true_labels = test_gene._targets.astype('float32')

        total_preds = y_pred_labels if total_preds is None else \
                      np.concatenate((total_preds, y_pred_labels), axis=0)
        total_trues = y_true_labels if total_trues is None else \
                      np.concatenate((total_trues, y_true_labels), axis=0)
        total_fnames += test_gene.filenames

        results = model.evaluate_generator(
            gene_wrapper(test_gene), 
            steps=tst_data.shape[0] / BATCH_SIZE)

        exp_results = pd.DataFrame.from_dict(
            {model.metrics_names[i]:[results[i]] for i in range(len(results))})
        exp_results.to_csv(
                OUT_PATH + 'fold{}.csv'.format(fold), 
                index=False)
        
        save_class_results(
            y_true_labels, 
            y_pred_labels, 
            test_gene.filenames,
            fold)

        trn_data, val_data, tst_data = roll_df(trn_data, val_data, tst_data, fold)

    save_class_results(
        total_trues,
        total_preds,  
        total_fnames, 
        "Total")

def save_class_results(y_groundtruth, y_predicted, file_names, fold):
    metrics = np.zeros((3, 9))
    metrics_names = [
        'Bal_Acc', 'Prec_Mic', 'Prec_Mac', 'Prec_Wgt', 
        'Rec_Mic', 'Rec_Mac', 'Rec_Wgt', 'Jac_Mac', 'Jac_Wgt']
    preds_df = None
    report_df = None
    for i, grade in enumerate(['BE', 'ICM', 'TE']):
        y_preds = np.argmax(y_predicted[:,:3], axis=1)
        y_truth = y_groundtruth[:,i].round() 

        metrics[i,0] = balanced_accuracy_score(y_truth, y_preds)
        metrics[i,1] = precision_score(y_truth, y_preds, average='micro')
        metrics[i,2] = precision_score(y_truth, y_preds, average='macro')
        metrics[i,3] = precision_score(y_truth, y_preds, average='weighted')
        metrics[i,4] = recall_score(y_truth, y_preds, average='micro')
        metrics[i,5] = recall_score(y_truth, y_preds, average='macro')
        metrics[i,6] = recall_score(y_truth, y_preds, average='weighted')
        metrics[i,7] = jaccard_score(y_truth, y_preds, average='macro')
        metrics[i,8] = jaccard_score(y_truth, y_preds, average='weighted')
        
        if grade == 'BE':
            y_truth = [REBMUN.get(item, item) for item in y_truth]
            y_preds = [REBMUN.get(item, item) for item in y_preds]
            classes = [4, 3, 2]
        else:
            y_truth = [RETTEL.get(item, item) for item in y_truth]
            y_preds = [RETTEL.get(item, item) for item in y_preds]
            classes = ['A', 'B', 'C']
        class_report_index = pd.Index(
            [classes[0], classes[1], classes[2],\
            'micro avg', 'macro avg', 'weighted avg'])

        cmat = confusion_matrix(y_truth, y_preds)
        out_name = OUT_PATH + 'CM_{}fold{}.png'.format(grade, fold)
        plot_confusion_matrix(
            cmat=cmat, 
            classes=classes, 
            out_name=out_name)

        pred_results = pd.DataFrame({
            "Filenames": file_names,
            "Labels" + grade: y_truth,
            "Preds" + grade: y_preds})

        preds_df = pred_results.set_index("Filenames") if preds_df is None else \
                   preds_df.join(pred_results.set_index("Filenames"))

        class_report = classification_report(y_truth, y_preds, output_dict=True)
        class_report = pd.DataFrame(class_report).transpose()
        class_report = class_report.set_index(class_report_index)
        report_df = class_report if report_df is None else \
                    report_df.append(class_report)
    
    preds_df.to_csv(
        OUT_PATH + "Preds-fold{}.csv".format(fold), 
        index=True)
    prerec_results = pd.DataFrame.from_dict(
        {metrics_names[i]:[metrics[0,i], metrics[1,i], metrics[2,i]] for i in range(9)}
    ).set_index(pd.Index(['BE', 'ICM', 'TE']))
    prerec_results.to_csv(
        OUT_PATH + "PrecRec-fold{}.csv".format(fold), 
        index=True)
    report_df.to_csv(
        OUT_PATH + "ClassificationReport-fold{}.csv".format(fold), 
        index=True)
 
def plot_confusion_matrix(cmat, classes, out_name):
    print(cmat)
    cmap = plt.cm.get_cmap('Blues')
    title = 'Confusion matrix, without normalization'

    fig, ax = plt.subplots()
    im = ax.imshow(cmat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cmat.shape[1]),
        yticks=np.arange(cmat.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cmat.max() / 2.
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(j, i, format(cmat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cmat[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_name)
    plt.close()

if __name__ == '__main__':
  test_()