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
