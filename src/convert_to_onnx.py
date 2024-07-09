import tensorflow as tf
import tf2onnx

# Load the Keras model
model = tf.keras.models.load_model('models/model.h5')

# Ensure the model is built and called at least once
sample_input = tf.random.uniform((1,) + model.input_shape[1:])
model(sample_input)

# Define the input signature
input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]
output_path = "models/model.onnx"

# Use the functional API to explicitly define inputs and outputs
inputs = tf.keras.Input(shape=model.input_shape[1:], name="input")
outputs = model(inputs)
functional_model = tf.keras.Model(inputs, outputs)

# Convert the Keras model to ONNX format
model_proto, _ = tf2onnx.convert.from_keras(functional_model, input_signature=input_signature, opset=13)

# Save the ONNX model
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
