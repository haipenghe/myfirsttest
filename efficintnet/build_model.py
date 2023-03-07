# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:11:23 2023

@author: HTHT
"""
from read_data import im_height,im_width
import efficientnet.tfkeras as efn
import tensorflow as tf
import time as t


#使用efficientnet.tfkeras的EfficientNetB0网络，并且使用官方的预训练模型
while True:
    try:
        covn_base = efn.EfficientNetB0(weights='imagenet', include_top=False ,input_shape=[im_height,im_width,3])
        break
    except:
        print("Connection refused by the server..")
        print("Let me sleep for 5 seconds")
        t.sleep(5)
        print("Was a nice sleep, now let me continue...")
        continue
covn_base.trainable = True

#冻结前面的层，训练最后10层
for layers in covn_base.layers[:-10]:
    layers.trainable = False
    
#构建模型  
model = tf.keras.Sequential([
        covn_base,
        tf.keras.layers.GlobalAveragePooling2D(), #加入全局平均池化层
        tf.keras.layers.Dense(10, activation='softmax') #添加输出层(10分类)
    ])
    
#打印每层参数信息 
model.summary()  

#编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(), #使用adam优化器
    loss = 'categorical_crossentropy', #交叉熵损失函数
    metrics=['accuracy'] #评价函数
)
