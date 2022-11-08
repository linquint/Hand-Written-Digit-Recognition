import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# load the data
(train_img,train_label),(test_img,test_label) = keras.datasets.mnist.load_data()
train_img = train_img.reshape([-1, 28, 28, 1])
test_img = test_img.reshape([-1, 28, 28, 1])
train_img = train_img/255.0
test_img = test_img/255.0

# convert class vectors to binary class matrices --> one-hot encoding
train_label = keras.utils.to_categorical(train_label)
test_label = keras.utils.to_categorical(test_label)

# data augmentation
shift = 0.15
datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.1, width_shift_range=shift, height_shift_range=shift)
datagen.fit(train_img)

# Learning rate reduction that will half the learning rate after validation accuracy decreases twice
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1], activation='relu'),
    keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1], activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (5, 5), padding="same", activation='relu'),
    keras.layers.Conv2D(64, (5, 5), padding="same", activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(
    datagen.flow(train_img, train_label, batch_size=256),
    validation_data=(test_img,test_label),
    epochs=20,
    callbacks=[learning_rate_reduction]
)
test_loss,test_acc = model.evaluate(test_img, test_label)
print('Test accuracy:', test_acc)

# save model as tfjs format
tfjs.converters.save_keras_model(model, 'models')