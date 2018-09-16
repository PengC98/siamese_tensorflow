import LeNet5_infernece
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image

# 1. 定义神经网络相关的参数
BATCH_SIZE = 50
lr = 0.00001
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 0
MODEL_SAVE_PATH = "E:/Python35/model/siamesemod" # 在当前目录下存在LeNet5_model子文件夹
MODEL_NAME = "siamese_model.ckpt"
INPUT_DATA = 'E:/facedata'
MODEL_DIR = 'E:/python35/mo'
MODEL_FILE= 'tensorflow_inception_graph.pb'
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
train=True

def create_image_lists(testing_percentage, validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG','png']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()

        # 初始化
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path
def get_or_create_bottleneck(sess, image_lists, label_name, index, category):
    label_lists = image_lists[label_name]

    image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)

    image_data = Image.open(image_path)
    image_data=np.array(image_data)


    return image_data
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category):
    bottlenecks = np.zeros([BATCH_SIZE, 64 ,64,3])
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        image_data = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category)
        
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks[_,:]=image_data
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths
def siamese_loss(out1,out2,y,Q=5):

    Q = tf.constant(Q, name="Q",dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2),1))   
    pos = tf.multiply(tf.multiply(y,2/Q),tf.square(E_w))
    neg = tf.multiply(tf.multiply(1-y,2*Q),tf.exp(-2.77/Q*E_w))                
    loss = pos + neg                 
    loss = tf.reduce_mean(loss)              
    return loss
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):

    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values
def inception_v3_siamese(x):
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
        # 定义新的神经网络输入
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

        # 定义一层全链接层
        #with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001), name="weights")
        biases = tf.Variable(tf.zeros([n_classes]), name="biases")
        logits = tf.matmul(bottleneck_input, weights) + biases
def siamese(x,keep_prob):
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
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
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
            fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

            fc1 = tf.nn.leaky_relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            if train: fc1 = tf.nn.dropout(fc1, 0.5)
        
        with tf.variable_scope('layer8-fc2'):
            fc2_weights = tf.get_variable("weight", [FC_SIZE, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
            fc2_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
            fc2 = tf.nn.leaky_relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
            if train: fc2 = tf.nn.dropout(fc1, 0.5)
        
        with tf.variable_scope('layer9-fc3'):
            fc3_weights = tf.get_variable("weight", [FC_SIZE, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
            fc3_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc2, fc3_weights) + fc3_biases
        return logit
if __name__=='__main__':
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    with tf.variable_scope('input_x1') as scope:
        x1 = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS])
    with tf.variable_scope('input_x2') as scope:
        x2 = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS])
    with tf.variable_scope('y') as scope:
        y = tf.placeholder(tf.float32, shape=[BATCH_SIZE])

    with tf.name_scope('keep_prob') as scope:
        keep_prob = tf.placeholder(tf.float32)
        
    with tf.variable_scope('siamese') as scope:
        out1 = siamese(x1,keep_prob)
        scope.reuse_variables()
        out2 = siamese(x2,keep_prob)
    with tf.variable_scope('metrics') as scope:
        loss = siamese_loss(out1, out2, y)
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    loss_summary = tf.summary.scalar('loss',loss)
    merged_summary = tf.summary.merge_all()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter('D:\python35\graph\siamese',sess.graph)

        for itera in range(TRAINING_STEPS):
            xs_1, ys_1 = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH_SIZE, 'training')
            xs_2, ys_2 = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH_SIZE, 'training')

            ys_1 = np.argmax(ys_1,axis=1)
            ys_2 = np.argmax(ys_2,axis=1)
            y_s = np.array(ys_1==ys_2,dtype=np.float32)
            _,train_loss,summ = sess.run([optimizer,loss,merged_summary],feed_dict={x1:xs_1,x2:xs_2,y:y_s,keep_prob:0.6})

            writer.add_summary(summ,itera)
            if itera % 10 == 0 :
                print('iter {},train loss {}'.format(itera,train_loss))
            if itera ==TRAINING_STEPS-1:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
                print("training over!")
                break
        validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH_SIZE, 'validation')
        embed = sess.run(out1,feed_dict={x1:validation_bottlenecks,keep_prob:0.6})
        writer.close()
