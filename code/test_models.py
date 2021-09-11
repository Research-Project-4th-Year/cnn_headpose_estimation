import os

architectures = ['Squeezenet_1_0', 'Squeezenet_1_1', 'ResNet34', 'ResNet18', 'MobileNetV2', 'SEResNet50', 'DenseNet201']
# architectures = ['Squeezenet_1_0']
time_list = []

for arch in architectures:

    print("-----------------"+arch+"----------------------")
    command = 'python test_hopenet.py --data_dir ./datasets/BIWI --filename_list ./datasets/BIWI/biwi_test.txt --snapshot output/snapshots/' + arch +'_Basic.pkl  --dataset BIWI --arch ' + arch
    os.system(command)
    