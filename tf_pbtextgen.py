# Generate pbtext file from pb file

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import os

#Define pb file path
pb_file_path = 'frozen_graph.pb'
#Define pbtext file path

pbtxt_file_path = 'frozen_graph.pbtxt'

#Read pb file

with gfile.FastGFile(pb_file_path,'rb') as f:

    graph_def = tf.compat.v1.GraphDef()

    graph_def.ParseFromString(f.read())

    g_in = tf.import_graph_def(graph_def)

#Write pbtext file

tf.io.write_graph(graph_def, '.', pbtxt_file_path, as_text=True)