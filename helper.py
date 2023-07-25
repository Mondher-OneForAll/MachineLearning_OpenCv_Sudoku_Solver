# Tensorflow to TFLite
import tensorflow as tf
from keras.models import load_model

# Converting the model
model = load_model('myModel.h5')
conveter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = conveter.convert()

# Saving the model
with open('myModel.tflite', 'wb') as f:
    f.write(tflite_model)
