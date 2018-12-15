# coding=utf8
from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
import time

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_weights#load_model

import matplotlib.pyplot as plt
#import keras.models
from keras.preprocessing import image
from keras_retinanet import models #使用这个来load
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

labels_to_names = {0: 'single', 1: 'red', 2: 'black'}

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/retinanet_1k_6.h5'
#MODEL_PATH = 'models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
# Load your trained model
model = models.load_model(MODEL_PATH, backbone_name='resnet50')
#model = load_model(MODEL_PATH) #使用这个即使是原来resnet50_weights_tf_dim_ordering_tf_kernels.h5也不能正常装载
# model._make_predict_function()          # Necessary??
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet') #取消自带的
print('Model loaded. Check http://127.0.0.1:5000/')

#model_predict是原来的分类的方式
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

#增加draw_out是自己模型
def Draw_out(path) :
   # load image
   image = read_image_bgr(path)
   # copy to draw ondraw = image.copy()
   draw = image.copy()
   draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

   # preprocess image for network
   image = preprocess_image(image)
   image, scale = resize_image(image)
   #print ('scale',scale)

   # process image
   start = time.time()
   boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
   #boxes是检测到的可能目标的框框
   #print ('boxes',boxes)
   #是对应的分值
   #print ('scores',scores)
   #对应的第几个标签
   #print ('labels',labels)
   #print("processing time: ", time.time() - start)

   # correct for image scale
   boxes /= scale

   # visualize detections
   for box, score, label in zip(boxes[0], scores[0], labels[0]):
       # scores are sorted so we can break
       if score < 0.4:
           break

       color = label_color(label)
       #print ('label',label)
       b = box.astype(int)
       #b是box取整
       #print ('b ',b)
       draw_box(draw, b, color=color)

       caption = "{} {:.3f}".format(labels_to_names[label], score)
       #single 0.431 是标签和分值
       #print ('caption ',caption)
       draw_caption(draw, b, caption)
   plt.figure(figsize=(10, 10))
   plt.axis('on')
   plt.imshow(draw)
   #plt.show()
   plt.savefig("static/js/powerbank_out.png")
   return caption

@app.route('/', methods=['GET'])  #这里是api的入口
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST']) #这里是api的入口
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = Draw_out(file_path)
        '''# Make prediction
        preds = model_predict(file_path, model) #上传了后即预测

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string'''
        return result  #将caption返回给Ajax，显示

    return None


if __name__ == '__main__':
    #app.run(port=5002, debug=True) #加上这个是什么结果是否5002是可以看到Debug信息？

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
