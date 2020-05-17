# Blastocyst-Grading
Keras implementation for the MMSP2019 paper: Multi-Label Classification for Automatic Human Blastocyst Grading with Severely Imbalanced Data

## Getting Started
### Installation
This code was run with python 3.6 and CUDA 9.0. To install libraries using conda,
```
conda install tensorflow-gpu==1.11 keras==2.2.4 scikit-learn scikit-image pandas xlrd ipykernel
```

### Training the network
The model can be trained by running: 
```
python train.py --train_name experiment_name --img_path path_to_image_directory --anno_file path_to_xlsx_annotation_file
```
The training process partitions a complete (single-directory) dataset into training, validation, and test sets and rolls the data for 3-fold cross validation.

### Testing the network
Similarly, a trained model can be tested by running: 
```
python test.py --train_name experiment_name --img_path path_to_image_directory --anno_file path_to_xlsx_annotation_file
```
Testing is performed on 3-folds separately and also the combined set. Class activation maps (CAMs) of each image are saved into a directory and show which areas in the image were active for each grade (e.g. below).
| Blastocyst Expansion Grade CAM| ICM Grade CAM |
|:---:|:---:|
| <img src="https://github.com/llockhar/Blastocyst-Grading/blob/master/demoImages/CAM_BE.jpg" alt="Blastocyst Expansion Grade CAM" width="250"/> | <img src="https://github.com/llockhar/Blastocyst-Grading/blob/master/demoImages/CAM_ICM.jpg" alt="ICM Grade CAM" width="250"/> |

## Background
### Blastocyst Grading System
Blastocysts (day-5 embryos) are assigned quality grades according to the Gardner grading system [1]. This score is used to determine which blastocyst in the treatment cycle batch has the highest likelihood of leading to a successful pregnancy. Below is an image of a blastocyst with labeled components and image artifacts that can interfere with image analysis algorithms. The pre-processing algorithm is available in this [repository](https://github.com/llockhar/Embryo-Image-Pre-processing). On the right is the Gardner grading system with scores present in the dataset.

| Blastocyst Components | Gardner Grading System |
|:---:|:---:|
| <img src="https://github.com/llockhar/Blastocyst-Grading/blob/master/demoImages/Labeled%20Embryo.png" alt="Blastocyst Components" width="350"/> | <img src="https://github.com/llockhar/Blastocyst-Grading/blob/master/demoImages/BlastocystGrades.png" alt="Gardner Grading System" width="350"/> |

### Network Architecture
The network has a VGG16 base with 3 output branches, one for each grade. Trainable convolutional kernel weights in the output branches are updated according to their respective label error, and weights in the backbone are updated as an average of all 3 label errors.

<img src="https://github.com/llockhar/Blastocyst-Grading/blob/master/demoImages/NetworkDiagram.png" alt="Network Diagram" width="400"/>

