import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('/home/nimnim/gunshot/gunshot2023summer/gunshot_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('gunshot_model.tflite', 'wb') as f:
    f.write(tflite_model)