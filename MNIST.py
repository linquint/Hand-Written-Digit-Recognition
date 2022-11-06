import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# load the data
(train_img,train_label),(test_img,test_label) = keras.datasets.mnist.load_data()
train_img = train_img.reshape([-1, 28, 28, 1])
test_img = test_img.reshape([-1, 28, 28, 1])
train_img = train_img/255.0
test_img = test_img/255.0
train_label = keras.utils.to_categorical(train_label)
test_label = keras.utils.to_categorical(test_label)

shift = 0.2
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=30, width_shift_range=shift, height_shift_range=shift)
datagen.fit(train_img)

# define the model architecture
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

# train the model
model.fit(
    datagen.flow(train_img, train_label, batch_size=32),
    validation_data=datagen.flow(test_img,test_label, batch_size=8),
    steps_per_epoch=len(train_img) / 32,
    epochs=5
)
test_loss,test_acc = model.evaluate(test_img, test_label)
print('Test accuracy:', test_acc)

# save model as tfjs format
tfjs.converters.save_keras_model(model, 'models')