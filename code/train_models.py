import os
import time

architectures = ['ResNet50','Squeezenet_1_0', 'Squeezenet_1_1', 'ResNet34', 'ResNet18', 'MobileNetV2', 'SEResNet50']
#architectures = ['ResNet50']
time_list = []

for arch in architectures:

    command = 'python train_hopenet.py --dataset BIWI --data_dir ./datasets/BIWI --lr 0.00001 --alpha 1 --batch_size 16 --num_epochs 100 --arch ' + arch
    
    start_time = time.time()
    os.system(command)
    train_time = time.time() - start_time
    print(arch + "train time: " + str(train_time))
    time_list.append(train_time)

print(time_list)
