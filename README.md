# Real-time Drowsiness Detection using TensorFlow Object Detection API
This is a part of the project of AASD 4014 - Deep Learning II. In this project, I am responsible to build an drowsiness detection model by TensorFlow Object Detection API. The detection model is trained on Google Colab and Google Drive. After training the model, it is downloaded to the local laptop and connect to the webcam for real-time drowsiness detection. 

## Problem Statement
Drowsy driving is one of the major cause of road accidents and causes 100,000 crashes & 1,550 deaths each year. To relief this problem, a real-time object detection system which can alert the driving system in a car when it detects a sleepy driver.

#### <b>Data - Drowsiness Prediction Dataset</b>
<font size=3>The link of Kaggle:</font>
https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset

#### <b>Reference - TRAINING AN OBJECT DETECTION MODEL FOR CUSTOM DATASET</b>
<font size=3>The link of reference:</font>
https://colab.research.google.com/drive/1QCU_dCR0ozI8j6X2btEDCsaUk5p_b1uw?usp=sharing#scrollTo=rHnTmrSwNg6S

## Annotate the Data on Roboflow

## Prepare the Workplace in Google Drive and Install TensorFlow API
- Object_Detection is the main folder of this project in Google Drive. Create a new folder called customTF2 in Object_Detection. 
<br><br>
- In customTF2, create 3 folders: data, pre-trained and training. 
<br><br>
- In data folder, upload the train_labels.csv, valid_labels.csv and test_labels.csv created by the Roboflow and create a folder called images and upload the train, valid and test sets.
<br><br>
- Clone the Tensorflow API into the main folder Object_Detection.

- Download the generate_tfrecord.py from [link](https://github.com/techzizou/Train-Object-Detection-Model-TF-2.x "link") and upload it to the customTF2 folder.


The structure of the workplace is as follows:
<img width="334" alt="structure" src="https://user-images.githubusercontent.com/101066418/230780555-777142a2-a494-48cc-9c1a-d48a3115a6e7.png">

After installation of Tensorflow API, we can start building the drowsiness detection model by running active_fatigued.ipynb in this repository on Colab.

## Evaluation: mAP & Total Loss

The mAP of the model at different IoU is as follows:

| IoU | mAP    |
| :---:   | :---: |
| 0.50:0.95 | 0.939 |
| 0.50 | 1.000  |
| 0.75 | 1.000  |

After training, Tensorboard is used to read the total loss of the model and the total loss at the final step (15000) is 0.0038405838.

<img width="1074" alt="Loss:total_loss" src="https://user-images.githubusercontent.com/101066418/230780665-d7d0359e-996d-4453-bb1d-1fd201f65ae5.png">

## Testing of the Model





