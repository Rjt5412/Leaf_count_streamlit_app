
import tensorflow as tf
import time

# Load the model
regression2_model_path = "DenseNet121_unquantized/"
densenet_tf_model = tf.keras.models.load_model(regression2_model_path)
# Load the quantized model
interpreter_quant = tf.lite.Interpreter(model_path="DenseNet121_quantized/optimized_densenet_model.tflite")
interpreter_quant.allocate_tensors()
input_index = interpreter_quant.get_input_details()[0]["index"]
output_index = interpreter_quant.get_output_details()[0]["index"]

def get_prediction_count_DenseNet121_transfer_learning_model(image):
    
    # Read the image path
    #image = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float64)
    # Resizing
    image = tf.image.resize(image, [256, 256])

    image = tf.reshape(image, [1, 256, 256, 3])
    
    #Unquantized model
    start_time = time.time()
    prediction = densenet_tf_model.predict(image)[0][0]
    prediction = round(prediction)
    time_taken = time.time() - start_time

    # For tflite models
    start_time = time.time()
    interpreter_quant.set_tensor(input_index, image)
    interpreter_quant.invoke()
    prediction_quantized = interpreter_quant.get_tensor(output_index)[0][0]
    time_taken_quantized = time.time() - start_time
    
    return prediction, round(prediction_quantized), time_taken, time_taken_quantized
