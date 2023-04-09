# Real-time Drowsiness Detection using TensorFlow Object Detection API

This is a part of the project of AASD 4014 - Deep Learning II. In this project, I am responsible to build an drowsiness detection model by TensorFlow Object Detection API. The detection model is trained on Google Colab and Google Drive. After training the model, it is downloaded to the local laptop and connect to the webcam for real-time drowsiness detection.

The model used is the Faster R-CNN ResNet101 V1 640x640 from [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md "model zoo")

## Problem Statement

Drowsy driving is one of the major cause of road accidents and causes 100,000 crashes & 1,550 deaths each year. To relief this problem, a real-time object detection system which can alert the driving system in a car when it detects a sleepy driver.

#### <b>Data - Drowsiness Prediction Dataset</b>

<font size=3>The link of Kaggle:</font>
https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset

#### <b>Reference - TRAINING AN OBJECT DETECTION MODEL FOR CUSTOM DATASET</b>

<font size=3>The link of reference:</font>
https://colab.research.google.com/drive/1QCU_dCR0ozI8j6X2btEDCsaUk5p_b1uw?usp=sharing#scrollTo=rHnTmrSwNg6S

## Annotate the Images on Roboflow

The annotation of the data is done on Roboflow. Upload the images downloaded from Kaggle to Roboflow and annotate the state of active and fatigued manually as follows:
<img width="1107" alt="anotation" src="https://user-images.githubusercontent.com/101066418/230782340-a1cd920a-d85a-4d3b-a500-182c587d6402.png">

After annotation, split the images into train, valid and test set and export them in TensorFlow Object Detection format.

<img width="508" alt="ratio" src="https://user-images.githubusercontent.com/101066418/230782753-59a90920-86f1-4d0e-9c98-92bfc3774f55.png">

After unzipping the file, there will be train, valid and test folders. In each folder, there is a .csv file recording the coordinate of the detection box of each image. Rename the .csv to train_labels.csv, valid_labels.csv and test_labels.csv. They will be used when building the detection model.

## Prepare the Workplace in Google Drive and Install TensorFlow API

- Object_Detection is the main folder of this project in Google Drive. Create a new folder called customTF2 in Object_Detection.
  <br><br>
- In customTF2, create 3 folders: data, pre-trained and training.
  <br><br>
- In data folder, upload the train_labels.csv, valid_labels.csv and test_labels.csv created by the Roboflow and create a folder called images and upload the train, valid and test sets.
  <br><br>
- Clone the Tensorflow API into the main folder Object_Detection.
<br><br>
- Download the generate_tfrecord.py from [link](https://github.com/techzizou/Train-Object-Detection-Model-TF-2.x "link") and upload it to the customTF2 folder.

The structure of the workplace is as follows:

<img width="321" alt="structure" src="https://user-images.githubusercontent.com/101066418/230784599-256f7fce-420f-4759-93db-e2a03c232aff.png">

After installation of Tensorflow API, we can start building the drowsiness detection model by running active_fatigued.ipynb in this repository on Colab.

## Evaluation: mAP & Total Loss

The mAP of the model at different IoU is as follows:

|    IoU    |  mAP  |
| :-------: | :---: |
| 0.50:0.95 | 0.939 |
|   0.50    | 1.000 |
|   0.75    | 1.000 |

After training, Tensorboard is used to plot the total loss of the model during training and the total loss at the final step (15000) is 0.0038405838.

<img width="1074" alt="Loss:total_loss" src="https://user-images.githubusercontent.com/101066418/230780665-d7d0359e-996d-4453-bb1d-1fd201f65ae5.png">

## Testing of the Model

Active case:

![active](https://user-images.githubusercontent.com/101066418/230783090-12d285a1-d066-4ca9-a9c8-acb45edbb086.png)

Fatigued case:

![fatugued](https://user-images.githubusercontent.com/101066418/230783096-08792c3c-7b69-4ba1-87d8-43212317a484.png)

## Real-time Drowsiness Detection

After training the model on Colab, the code of real-time detection must be run on local laptop.

This is the recording of using Real-time Drowsiness Detection: [video](https://github.com/ccmak514/drowsiness-detection/blob/main/Faster%20RCNN.mp4 "video")
