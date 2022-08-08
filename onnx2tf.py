# import library
from onnx_tf.backend import prepare
import onnx

# set parameter
onnx_model_path = 'weight.onnx'
tf_model_path = 'weights_tf'

# convert model tensorflow
onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

# ------------------- check weight tf -------------------- #
# import library
import tensorflow as tf

# set parameter
tf_model_path = 'weights_tf'

# load weight tensorflow
model = tf.saved_model.load(tf_model_path)
model.trainable = False

# create input and check model
input_tensor = tf.random.uniform([1, 512, 512, 3])
out = model(**{'input': input_tensor})
print(out)
