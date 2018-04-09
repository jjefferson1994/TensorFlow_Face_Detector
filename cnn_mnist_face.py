from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import glob
import cv2
import os

tf.logging.set_verbosity(tf.logging.INFO)

########################################################################################################################
#retrieves pictures from file path to save on computing
#gets training set from specified path
########################################################################################################################
def getTrainingList():
    faceList = glob.glob('C:/Users/John Jefferson/PycharmProjects/Proj2/trainingSet/faces/*.jpg')
    backList = glob.glob('C:/Users/John Jefferson/PycharmProjects/Proj2/trainingSet/backgrounds/*.jpg')
    return (faceList, backList)

########################################################################################################################
#retreives pictures from file path to save on computing
#gets training set from specified path
########################################################################################################################
def getTestList():
    faceList = glob.glob('C:/Users/John Jefferson/PycharmProjects/Proj2/testSet/faces/*.jpg')
    backList = glob.glob('C:/Users/John Jefferson/PycharmProjects/Proj2/testSet/backgrounds/*.jpg')
    return (faceList, backList)

########################################################################################################################
#generates a training set and saves to specified path
########################################################################################################################
def generateTrainingSet(numOfPictures,xydim):
    pictlist = glob.glob('orig_images/*/*.jpg') # generate a list of all the files with pictures
    dim = (xydim,xydim) # used for testing purposes, can be changed to change averaging range and avg blank size
    facePath = 'C:/Users/John Jefferson/PycharmProjects/Proj2/trainingSet/faces'
    backPath = 'C:/Users/John Jefferson/PycharmProjects/Proj2/trainingSet/backgrounds'

    # for loop represents the number of face images to be averaged
    for i in range(0, numOfPictures):
        currentPict = (pictlist[i])
        img = cv2.imread(currentPict, 1)

        imgFaceCropped = img[60:200,75:175] #crops the image around the face (dataset specific)
        imgBackCropped = img[0:75,0:75]

        resizedFace = cv2.resize(imgFaceCropped,
            dim, interpolation=cv2.INTER_AREA) #resizes the images chosen]
        resizedBack = cv2.resize(imgBackCropped,
            dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(facePath,
            '_0' + str(i) + '.jpg'),
            resizedFace)
        cv2.imwrite(os.path.join(backPath,
            '_0' + str(i) + '.jpg'),
            resizedBack)
    return

########################################################################################################################
#generates a test set from diffrent pictures than the training one
########################################################################################################################
def generateTestSet(trainingNum,numOfPictures,xydim):
    pictlist = glob.glob('orig_images/*/*.jpg') # generate a list of all the files with pictures
    dim = (xydim,xydim) # used for testing purposes, can be changed to change averaging range and avg blank size
    facePath = 'C:/Users/John Jefferson/PycharmProjects/Proj2/testSet/faces'
    backPath = 'C:/Users/John Jefferson/PycharmProjects/Proj2/testSet/backgrounds'

    # for loop represents the number of face images to be averaged
    for i in range(trainingNum, trainingNum + numOfPictures):
        currentPict = (pictlist[i])
        img = cv2.imread(currentPict, 1)

        imgFaceCropped = img[60:200, 75:175] #crops the image around the face (dataset specific)
        imgBackCropped = img[0:75, 0:75]

        resizedFace = cv2.resize(imgFaceCropped,
            dim, interpolation=cv2.INTER_AREA) #resizes the images chosen]
        resizedBack = cv2.resize(imgBackCropped,
            dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(facePath,
            '_0' + str(i) + '.jpg'),
            resizedFace)
        cv2.imwrite(os.path.join(backPath,
            '_0' + str(i) + '.jpg'),
            resizedBack)
    return

########################################################################################################################
#Add the application logic here
########################################################################################################################
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 52, 52, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 13 * 13 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

########################################################################################################################
#this runs the training model given from Google
########################################################################################################################
def runTrainingModel(trainingData,testData,trainingLabels,testLabels):
    # Load training and eval data
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = np.float32(trainingData.T) #mnist.train.images # Returns np.array
    train_labels = np.asarray(trainingLabels, dtype=np.int32) #np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = np.float32(testData.T) #mnist.test.images # Returns np.array
    eval_labels = np.asarray(testLabels,dtype=np.int32) #np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/epoch/proj2")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=10,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2500,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

########################################################################################################################
#This is the preprocessing step
########################################################################################################################
def getImages(dim):
    #generates facial images from data set and saves in file location (only needs one time run)
    #generateTrainingSet(10000, dim)
    #generateTestSet(5000, 1000, dim) #number of training images, number of test images, dimension

    #gets images for training and testing the models
    [faceTrainingSet, backTrainingSet] = getTrainingList()
    [faceTestSet, backTestSet] = getTestList()

    return [faceTrainingSet, backTrainingSet,faceTestSet, backTestSet]

########################################################################################################################
#Flattens and concatenates the image set passed inS
########################################################################################################################
def flattenImages(trainingSet):
    dim = len(cv2.imread(trainingSet[1])) #dimension of picture set
    numOfPictures = len(trainingSet)
    flattenedMatrix = np.zeros((dim*dim,numOfPictures))

    #flatten the images and add to larger matrix
    for i in range(0, numOfPictures):
        origImage = cv2.imread(trainingSet[i], 0) #0 parameter reads in as grayscale
        flatImage = origImage.flatten()

        #pushes flattened image into larger flattened image matrix
        flattenedMatrix.T[i] = flatImage

    return flattenedMatrix

########################################################################################################################
#Flattens and concatenates the image set passed in
#!!!!!!!!!!!figure out mean and std!!!!!!!!!!!!!!
########################################################################################################################
def standardizeImages(flatMatrix):
    meanOfData = 0 #np.mean(flatMatrix.T,axis = 0)
    stdOfData = 1 #np.mean(flatMatrix.T,axis=0)
    standardMatrix = (flatMatrix.T - meanOfData)/stdOfData
    #standardMatrix = #tf.nn.l2_normalize(flatMatrix, [0,1])

    return standardMatrix.T

########################################################################################################################
#Shapes the dataset into a complete form of flattened faces and backgrounds in one matrix
########################################################################################################################
def shapeDataset(dataset1, dataset2):
    xdim = len(dataset1)
    ydim = len(dataset1.T) + len(dataset2.T)
    completeDataset = np.zeros((xdim,ydim))

    for i in range(0,len(dataset1.T)):
        completeDataset.T[i] = dataset1.T[i]
    for i in range(len(dataset1.T),ydim):
        completeDataset.T[i] = dataset2.T[i - len(dataset1.T)]

    return completeDataset

########################################################################################################################
#Generates the labels of the datasets assuming first half is a  success
########################################################################################################################
def generateLabels(dataset):
    numOfPictures = len(dataset.T)
    labelSet = np.zeros(numOfPictures)
    for i in range(0,int(numOfPictures/2)):
        labelSet[i] = labelSet[i] + 1 #creates an array half full of ones and zeros

    return labelSet

########################################################################################################################
#This is the main function where the code runs
########################################################################################################################
def main(unused_argv):
    #gets the images for training and testing (constructs directory full of 60x60 gray scale images)
    [faceTrainingList, backTrainingList, faceTestList, backTestList] = getImages(52) #input results in NxN image

    #flattens all of the images into one matrix
    flatFaces = flattenImages(faceTrainingList)
    flatBackgrounds = flattenImages(backTrainingList)
    flatFacesTest = flattenImages(faceTestList)
    flatBackgroundTest = flattenImages(backTestList)

    #standardizes the images by subracting mean and dividing by std
    standardFlatFaces = standardizeImages(flatFaces)
    standardFlatBackgrounds = standardizeImages(flatBackgrounds)

    #shapes datasets into usable form by tensorflow
    trainingData = shapeDataset(standardFlatFaces,standardFlatBackgrounds)
    testData = shapeDataset(flatFacesTest,flatBackgroundTest)

    #generate labels matrix
    trainingLabels = generateLabels(trainingData) #assumes first half of data is a success
    testLabels = generateLabels(testData)

    #runs the tutorial training and test model
    runTrainingModel(trainingData,testData,trainingLabels,testLabels)

if __name__ == "__main__":
    tf.app.run()
