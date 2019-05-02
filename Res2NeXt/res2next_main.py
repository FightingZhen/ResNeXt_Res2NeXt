import sys

sys.path.append('../Data_Initialization/')
from res2next_model import Res2NeXt
from cifar100 import *
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True)
parser.add_argument('-gpu', required=True)
parser.add_argument('-cardinality', required=True, type=int)
parser.add_argument('-width', required=True, type=int)
parser.add_argument('-scale', required=True, type=int)
parser.add_argument('-se', required=True)

parser.add_argument('-epoch', default=300, type=int)
parser.add_argument('-num_class', default=100, type=int)
parser.add_argument('-ksize', default=3, type=int)
parser.add_argument('-weight_decay', default=5e-4, type=float)
parser.add_argument('-momentum', default=0.9, type=float)
parser.add_argument('-block_num1', default=3, type=int)
parser.add_argument('-block_num2', default=3, type=int)
parser.add_argument('-block_num3', default=3, type=int)
parser.add_argument('-learning_rate', default=0.1, type=float)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

train_data, test_data = tf.keras.datasets.cifar100.load_data()
train_x, train_y = train_data
test_x, test_y = test_data

train_x, test_x = color_preprocessing(train_x, test_x)

train_y = onehotEncoder(train_y, num_class=args.num_class)
test_y = onehotEncoder(test_y, num_class=args.num_class)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    resnext_model = Res2NeXt(model_name=args.model_name,
                             sess=sess,
                             train_data=[train_x, train_y],
                             tst_data=[test_x, test_y],
                             epoch=args.epoch,
                             num_class=args.num_class,
                             ksize=args.ksize,
                             weight_decay=args.weight_decay,
                             momentum=args.momentum,
                             cardinality=args.cardinality,
                             width=args.width,
                             scale=args.scale,
                             block_num1=args.block_num1,
                             block_num2=args.block_num2,
                             block_num3=args.block_num3,
                             learning_rate=args.learning_rate,
                             batch_size=args.batch_size,
                             img_height=args.img_height,
                             img_width=args.img_width,
                             use_se=args.se)
    resnext_model.train()
