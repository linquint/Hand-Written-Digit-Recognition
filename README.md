# Hand Written Digit Recognition
 Hand Written Digit Recognition using javascript library TensorFlow.js

## Changes Made to This Fork
 * Implemented data augmentation
 * Implemented learning rate reduction
 * Made some changes to the model architecture

After 20 epochs this model reached 99.68% of accuracy using the test dataset.
 
## Live Demo
Live demo for this fork:
**[https://hand-written-digit-recognition.pages.dev/](https://hand-written-digit-recognition.pages.dev/)**

Live demo for the original repo:
**[https://bensonruan.com/handwritten-digit-recognition-with-tensorflow-js/](https://bensonruan.com/handwritten-digit-recognition-with-tensorflow-js/)**

![handwritten-recognition](https://bensonruan.com/wp-content/uploads/2019/09/handwritten-recognition-5.gif)
 
## Installing
Clone this repository to your local computer
```bash
git clone https://github.com/bensonruan/Hand-Written-Digit-Recognition.git
```
Point your localhost to the cloned root directory

Browse to http://localhost/index.html  

## Start Predicting Hand Written Digit
* Draw on the canvas with your mouse on desktop or your finger on your mobile
* Click "Predict" to get result of the hand written digit prediction
* Click "Clean" to start drawing again

## Pre-trained model 
Use MNIST dataset from Keras with CNN (Convolutional Neural Network)
```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64, (5, 5), padding="same"),    
    keras.layers.MaxPool2D((2,2)),    
    keras.layers.Flatten(),   
    keras.layers.Dense(1024, activation='relu'),    
    keras.layers.Dropout(0.2),   
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Library
* [jquery](https://code.jquery.com/jquery-3.3.1.min.js) - JQuery
* [tensorflowjs](https://github.com/tensorflow/tfjs) - JavaScript library for training and deploying machine learning models
* [Chart.js](https://github.com/chartjs/Chart.js) - JavaScript library for display charts
