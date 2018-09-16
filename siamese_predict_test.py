import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import cv2
# 1. 定义神经网络相关的参数
BATCH_SIZE = 1
lr = 0.00001
#REGULARIZATION_RATE = 0.00001
TRAINING_STEPS = 5000
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 0
MODEL_SAVE_PATH = "E:/Python35/model/siamesemod" # 在当前目录下存在LeNet5_model子文件夹
MODEL_NAME = "siamese_model.ckpt"
INPUT_DATA = 'E:/facedata'
INPUT_NODE = 4096
OUTPUT_NODE = 3

IMAGE_SIZE = 64
NUM_CHANNELS = 3
NUM_LABELS = 2

CONV1_DEEP = 8
CONV1_SIZE = 3

CONV2_DEEP = 16
CONV2_SIZE = 3

CONV3_DEEP = 32
CONV3_SIZE = 3

FC_SIZE = 4000
train=False
def siamese(x,keep_prob):
        #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable(
                "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
            relu = tf.nn.leaky_relu(tf.nn.bias_add(conv1, conv1_biases))

        with tf.name_scope("layer2-pool1"):
            pool1 = tf.nn.max_pool(relu, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
        with tf.variable_scope("layer3-conv2"):
            conv2_weights = tf.get_variable(
                "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
        with tf.variable_scope("layer5-conv3"):
            conv3_weights = tf.get_variable(
                "weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu3 = tf.nn.leaky_relu(tf.nn.bias_add(conv3, conv3_biases))

        with tf.name_scope("layer6-pool2"):
            pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_shape = pool3.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool3, [pool_shape[0], nodes])

        with tf.variable_scope('layer7-fc1'):
            fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))
            #if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
            fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

            fc1 = tf.nn.leaky_relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            if train: fc1 = tf.nn.dropout(fc1, 0.5)
        
        with tf.variable_scope('layer8-fc2'):
            fc2_weights = tf.get_variable("weight", [FC_SIZE, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            #if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
            fc2_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
            fc2 = tf.nn.leaky_relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
            if train: fc2 = tf.nn.dropout(fc1, 0.5)
        
        with tf.variable_scope('layer9-fc3'):
            fc3_weights = tf.get_variable("weight", [FC_SIZE, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            #if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
            fc3_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc2, fc3_weights) + fc3_biases
        return logit


def predict(x,y):
    cv2.namedWindow("test")
    cap=cv2.VideoCapture(1)
    classfier=cv2.CascadeClassifier("E:/BaiduNetdiskDownload/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml")
    success,frame=cap.read()
    color=(0,255,0)
    with tf.variable_scope('input_x1') as scope:
        x1 = tf.placeholder(tf.float32, [
                BATCH_SIZE,
                64,
                64,
                3])
    with tf.variable_scope('input_x2') as scope:
        x2 = tf.placeholder(tf.float32, [
                BATCH_SIZE,
                64,
                64,
                3])
    validate_feed={x1:x,
                    x2:y}
    with tf.variable_scope('siamese') as scope:
        out1=siamese(x1,1)
        scope.reuse_variables()
        out2=siamese(x2,1)
    #E_w=tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(out1,out2),2),1))
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2),1))   
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))

        while success:
            success,frame=cap.read()
            size=frame.shape[:2]
            image=np.zeros(size,dtype=np.float16)
            image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)
            divisor=6
            h, w = size
            minSize =(w//divisor, h//divisor)
            faceRects = classfier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE,minSize)
            if len(faceRects)>0:
                for faceRect in faceRects: 
                    a, b, w, h = faceRect
                    cv2.rectangle(frame, (a, b), (a+w, b+h), color)
                    img=frame[b:b+h,a:a+w]
                    img=cv2.resize(img,(64,64))
                    img=np.array(img)
                    values=[]
                    values.append(img)
                    a=out1.eval(feed_dict={x1:values})
                    b=out2.eval(feed_dict={x2:x})
                    c=out2.eval(feed_dict={x2:y})
                    E_w1 = tf.sqrt(tf.reduce_sum(tf.square(a-b),1))
                    E_w2 = tf.sqrt(tf.reduce_sum(tf.square(a-c),1))
                    print("E_w1:")
                    print(E_w1.eval())
                    print("E_w2:")
                    print(E_w2.eval())
            cv2.imshow("test", frame)
            key=cv2.waitKey(10)
            c = chr(key & 255)
            if c in ['q', 'Q', chr(27)]: break
    cv2.destroyWindow("test")
img1=cv2.imread("E:/facedata/pc/500.jpg")
img2=cv2.imread("E:/facedata/dyz/10.jpg")
img1=np.array(img1)
img2=np.array(img2)
value1=[]
value1.append(img1)
value2=[]
value2.append(img2)
predict(value1,value2)


    
    
