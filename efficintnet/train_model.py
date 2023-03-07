# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:14:10 2023

@author: HTHT
"""
import matplotlib.pyplot as plt
from read_data import epochs,train_data_gen,total_train,batch_size,val_data_gen,total_val
from build_model import model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

#学习率设置
def lrfn(epoch):
    LR_START = 0.00001 #初始学习率
    LR_MAX = 0.0004 #最大学习率
    LR_MIN = 0.00001 #学习率下限
    LR_RAMPUP_EPOCHS = 5 #上升过程为5个epoch
    LR_SUSTAIN_EPOCHS = 0 #学习率保持不变的epoch数
    LR_EXP_DECAY = .8 #指数衰减因子
    
    if epoch < LR_RAMPUP_EPOCHS: #第0-第5个epoch学习率线性增加
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS: #不维持
        lr = LR_MAX
    else: #第6-第15个epoch学习率指数下降
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

#绘制学习率曲线
rng = [i for i in range(epochs)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
#print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


#训练设置
#使用tensorflow中的回调函数LearningRateScheduler设置学习率
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

#保存最优模型
checkpoint = ModelCheckpoint(
                                filepath='./save_weights/myefficientnet.ckpt', #保存模型的路径
                                monitor='val_acc',  #需要监视的值
                                save_weights_only=False, #若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
                                save_best_only=True, #当设置为True时，监测值有改进时才会保存当前的模型
                                mode='auto',  #当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min，在auto模式下，评价准则由被监测值的名字自动推断
                                period=1 #CheckPoint之间的间隔的epoch数
                            )
               
#开始训练
history = model.fit(x=train_data_gen,   #输入训练集
                    steps_per_epoch=total_train // batch_size, #一个epoch包含的训练步数
                    epochs=epochs, #训练模型迭代次数
                    validation_data=val_data_gen,  #输入验证集
                    validation_steps=total_val // batch_size, #一个epoch包含的训练步数
                    callbacks=[checkpoint, lr_schedule]) #执行回调函数

#保存训练好的模型权重                    
model.save_weights('./save_weights/myefficientnet.ckpt',save_format='tf')

# 记录训练集和验证集的准确率和损失值
history_dict = history.history
train_loss = history_dict["loss"] #训练集损失值
train_accuracy = history_dict["accuracy"] #训练集准确率
val_loss = history_dict["val_loss"] #验证集损失值
val_accuracy = history_dict["val_accuracy"] #验证集准确率

