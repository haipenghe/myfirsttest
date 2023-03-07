# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:24:44 2023

@author: HTHT
"""

#导入相应的库
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#设置图片的高和宽，一次训练所选取的样本数，迭代次数
im_height = 224
im_width = 224
batch_size = 128
epochs = 15

# 创建保存模型的文件夹
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")


image_path = os.getcwd() # 猫狗数据集路径
train_dir = image_path + "/training/training" #训练集路径
validation_dir = image_path + "/validation/validation" #验证集路径

# 定义训练集图像生成器，并进行图像增强
train_image_generator = ImageDataGenerator( rescale=1./255, # 归一化
                                            rotation_range=40, #旋转范围
                                            width_shift_range=0.2, #水平平移范围
                                            height_shift_range=0.2, #垂直平移范围
                                           shear_range=0.2, #剪切变换的程度
                                            zoom_range=0.2, #剪切变换的程度
                                            horizontal_flip=True,  #水平翻转
                                            fill_mode='nearest')
                                            
# 使用图像生成器从文件夹train_dir中读取样本，对标签进行one-hot编码
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir, #从训练集路径读取图片
                                                           batch_size=batch_size, #一次训练所选取的样本数
                                                           shuffle=True, #打乱标签
                                                           target_size=(im_height, im_width), #图片resize到224x224大小
                                                           class_mode='categorical') #one-hot编码
                                                           
# 训练集样本数        
total_train = train_data_gen.n 

# 定义验证集图像生成器，并对图像进行预处理
validation_image_generator = ImageDataGenerator(rescale=1./255) # 归一化

# 使用图像生成器从验证集validation_dir中读取样本
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,#从验证集路径读取图片
                                                              batch_size=batch_size, #一次训练所选取的样本数
                                                              shuffle=False,  #不打乱标签
                                                              target_size=(im_height, im_width), #图片resize到224x224大小
                                                              class_mode='categorical') #one-hot编码
                                                              
# 验证集样本数      
total_val = val_data_gen.n

