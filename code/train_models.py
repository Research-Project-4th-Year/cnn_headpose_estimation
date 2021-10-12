import os
import time
import torch

#architectures = ['Squeezenet_1_0', 'Squeezenet_1_1', 'ResNet34', 'ResNet18', 'MobileNetV2', 'SEResNet50', 'DenseNet201']
architectures = ['ResNet18']
time_list = []

for arch in architectures:

    #Train with RKD Loss
    command = 'python train_hopenet.py --dataset BIWI --data_dir ./datasets/BIWI --lr 0.00001 --alpha 1 --batch_size 16 --num_epochs 100  --arch ' + arch

    #Train with Dynamic KD Loss
    #command = 'python train_hopenet.py --dataset BIWI --data_dir ./datasets/BIWI --lr 0.00001 --alpha 1 --batch_size 64 --num_epochs 50 --w_dist 25.0 --w_angle 50.0  --kd_alpha_dynamic True --arch ' + arch

    #Train with exchange student and teacher loss
    #command = 'python train_hopenet.py --dataset BIWI --data_dir ./datasets/BIWI --lr 0.00001 --alpha 1 --batch_size 64 --num_epochs 50 --w_dist 25.0 --w_angle 50.0 --change_t_s True  --arch ' + arch

    #Train with Wasserstein Distance
    #command = 'python train_hopenet.py --dataset BIWI --data_dir ./datasets/BIWI --lr 0.00001 --alpha 1 --batch_size 64 --num_epochs 100 --kd_loss ws --arch ' + arch
   
    start_time = time.time()
    os.system(command)
    train_time = time.time() - start_time
    print(arch + "train time: " + str(train_time))
    time_list.append(train_time)
    #torch.cuda.empty_cache()

print(time_list)
