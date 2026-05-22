import tensorflow as tf
import seaborn as sns
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix


import matplotlib.pyplot as plt
import math

# ----------------- LOADING THE DATA  --------------
train_generator = ImageDataGenerator(
  rescale=1./255,             # normalize pixel values to 0-1
  rotation_range=15,
  width_shift_range=0.1,
  height_shift_range=0.1,
  zoom_range=0.1,
  horizontal_flip=True,    # chest X-rays can be flipped
  fill_mode='nearest'
)

test_generator = ImageDataGenerator(rescale=1./255)

# create training and validation iterators
training_iterator = train_generator.flow_from_directory(
  'augmented-data/train',
  target_size=(128, 128),
  class_mode='sparse',
  color_mode='grayscale',
  batch_size=5)


validation_iterator = test_generator.flow_from_directory(
  'augmented-data/test',
  target_size=(128, 128),
  class_mode='sparse',
  color_mode='grayscale',
  batch_size=25)

# ----------------- BUILD THE MODEL  --------------
model = tf.keras.Sequential()

# Input
model.add(tf.keras.Input(shape=(128, 128, 1)))

# block 1
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# block 2
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# block 3
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# transition to classification
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.3))

# hidden dense layer
model.add(tf.keras.layers.Dense(64, activation='relu'))

# output
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=[
    tf.keras.metrics.SparseCategoricalAccuracy()
  ]
)

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
  training_iterator,
  steps_per_epoch=training_iterator.samples/5,
  epochs=25,
  validation_data=validation_iterator,
  validation_steps=validation_iterator.samples/5,
  callbacks=[es]
)

# ----------------- PLOTTING  --------------
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['sparse_categorical_accuracy'])
ax1.plot(history.history['val_sparse_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='validation')
ax2.set_title('model loss')
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.legend(['train', 'validation'], loc='upper left')

# used to keep plots from overlapping
fig.tight_layout()
fig.savefig('static/images/my_plots.png')


# ----------------- EVALUATION  --------------
test_steps_per_epoch = math.ceil(validation_iterator.samples / validation_iterator.batch_size)

predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))

cm = confusion_matrix(true_classes, predicted_classes)
print(cm)


# visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('static/images/confusion_matrix.png')
plt.close()
