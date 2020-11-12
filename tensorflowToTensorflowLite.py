import tensorflow as tf
from tensorflow import lite
from tensorflow.keras.models import load_model

new_model = load_model('fnf.h5')

converter = lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

with open('fnflite.tflite', 'wb') as f:
  f.write(tflite_model)