#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
from io import BytesIO
import sys
import os
import tempfile

MODEL_BASE = '/opt/models/research'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')

#from decorator import requires_auth
from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask_wtf.file import FileField
import numpy as np
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
from utils import label_map_util
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError
import urllib.request

THRESHOLD = float(os.environ.get('THRESHOLD', '0.9'))
PORT = int(os.environ.get('PORT', '5000'))
DEBUG = os.environ.get('DEBUG', False)
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', 1))
if DEBUG != False:
  DEBUG = True

app = Flask(__name__)


@app.before_request
#@requires_auth
def before_request():
  pass


PATH_TO_CKPT = '/opt/graph_def/frozen_inference_graph.pb'
PATH_TO_LABELS = '/opt/configs/data/santa_label_map.pbtext'

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())

print(PATH_TO_LABELS)
def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image


class PhotoForm(Form):
  input_photo = FileField(
      'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
      validators=[is_image()])


class ObjectDetector(object):

  def __init__(self):
    self.detection_graph = self._build_graph()
    self.sess = tf.Session(graph=self.detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

  def _build_graph(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
  
  def custom_detect(self, image):
    # Definite input and output Tensors for detection_graph
    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = self._load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = self.sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores, classes.astype(int), num

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    graph = self.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = self.sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])

    return boxes, scores, classes.astype(int), num_detections


def draw_bounding_box_on_image(image, box, color='red', thickness=4):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  ymin, xmin, ymax, xmax = box
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)


def encode_image(image):
  image_buffer = BytesIO()
  image.save(image_buffer, format='PNG')
  #imgstr = 'data:image/png;base64,{!s}'.format(
  #    base64.b64encode(image.read()))
  imgstr = 'data:image/png;base64,'
  image_base64_str = base64.b64encode(image_buffer.getvalue())
  imgstr = imgstr + image_base64_str.decode('utf-8')
  return imgstr


def detect_objects(image_path):
  #image = Image.open(image_path).convert('RGB')
  #boxes, scores, classes, num_detections = client.detect(image)
  image = Image.open(image_path)
  boxes, scores, classes, num_detections  = client.custom_detect(image)
  image.thumbnail((480, 480), Image.ANTIALIAS)

  new_images = {}
  print(num_detections)
  print(scores)
  print(classes)

  for i in range(int(num_detections[0])):
    if scores[0][i] <= THRESHOLD: continue
    cls = classes[0][i]
    if cls not in new_images.keys():
      new_images[cls] = image.copy()
    draw_bounding_box_on_image(new_images[cls], boxes[0][i],
                               thickness=int(scores[0][i]*10)-4)

  result = {}
  result['original'] = encode_image(image.copy())

  for cls, new_image in new_images.items():
    category = client.category_index[cls]['name']
    result[category] = encode_image(new_image)

  return result


@app.route('/')
def upload():
  photo_form = PhotoForm(request.form)
  return render_template('upload.html', photo_form=photo_form, result={})


@app.route('/post', methods=['GET', 'POST'])
def post():
  form = PhotoForm(CombinedMultiDict((request.files, request.form)))
  url = request.form['url']
  if url != None:
      print('the url', url)

  if request.method == 'POST' and (form.validate() or url != None):
    
    try:
      if url != None:
        file_name = '/tmp/' + url.split('/')[-1]
        urllib.request.urlretrieve(url, file_name)
        result = detect_objects(file_name)
      else:
        with tempfile.NamedTemporaryFile() as temp:
          form.input_photo.data.save(temp)
          temp.flush()
          result = detect_objects(temp.name)

      photo_form = PhotoForm(request.form)
      return render_template('upload.html',
                            photo_form=photo_form, result=result)
    except:
      return redirect(url_for('upload'))
  else:
    return redirect(url_for('upload'))


client = ObjectDetector()


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
