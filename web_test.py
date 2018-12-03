# -*- coding:utf-8 -*-
import os
import argparse
from flask import Flask
from flask import request
import tensorflow as tf
import tensorflow.contrib.keras as kr
from cnn_model import TCNNConfig, TextCNN
from data_load.load import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

app = Flask(__name__)


def load_graph(frozen_graph_file):
    with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
        return graph


@app.route('/', methods=['POST', 'GET'])
def about():
    if request.method == "POST":
        message = request.form.get('message')
        # message = str(message)
        content = unicode(message)
        data = [word_to_id[x] for x in content if x in word_to_id]
        feed_dict = {
            input_x: kr.preprocessing.sequence.pad_sequences([data], config.seq_length),
            keep_prob: 1.0
        }
        out = persistent_sess.run(y, feed_dict=feed_dict)
	return str((categories[out[0]]))
    else:
        return """<form action="/" method="POST">
                  <input type="text" name="message" placeholder="请输入查询语句">
                  <input type="submit" value="Submit" name="ok"/>
                  </form>"""



if __name__ == "__main__":

    config = TCNNConfig()
    categories, cat_to_id = read_category()
    base_dir = 'data_solve'
    vocab_dir = os.path.join(base_dir, 'data.vocab.txt')
    words, word_to_id = read_vocab(vocab_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="checkpoints/textcnn/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()

    print('Loading the model')
    graph = load_graph(args.frozen_model_filename)
    input_x = graph.get_tensor_by_name('prefix/input_x:0')
    keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')
    y = graph.get_tensor_by_name('prefix/score/predict:0')

    # for op in graph.get_operations():
    #     print(op.name)

    persistent_sess = tf.Session(graph=graph)
    print('Starting the API')
    app.run(host='0.0.0.0', port=5000)
