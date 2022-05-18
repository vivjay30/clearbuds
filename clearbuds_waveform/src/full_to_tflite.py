import torch
import onnx
import torch
import tensorflow as tf
from onnx_tf.backend import prepare

# torch - 1.6.0
# torchvision - 0.7.0
# onnx - 1.7.0
# tensorflow - 2.2.0
# tensorflow-addons - 0.11.2
# onnx-tf - 1.8.0


# File names ( change as needed )
for name in ["ConvTasNet"]:
    onnx_model_name ='{}.onnx'.format(name)
    tf_model_name = '{}_pb'.format(name)
    tf_lite_model_name = '{}.tflite'.format(name)

    # Load the ONNX file
    model = onnx.load(onnx_model_name)

    # ONNX model to Tensorflow
    tf_rep = prepare(model)

    #Tensorflow Model 
    tf_rep.export_graph(tf_model_name)

    # TF to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_name)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    #Saving tflite model
    with tf.io.gfile.GFile(tf_lite_model_name, 'wb') as f:
        f.write(tflite_model)