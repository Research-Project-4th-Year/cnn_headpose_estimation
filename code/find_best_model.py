import os

for i in range(1,26):
    print("------------")
    print(f"model_no:{i}")
    #command = 'python test_hopenet.py --data_dir ./datasets/AFLW2000 --filename_list ./datasets/AFLW2000/files.txt --snapshot output/snapshots/ResNet50_epoch_'+str(i)+'.pkl  --dataset AFLW2000 --arch ResNet50'
    #command = 'python test_hopenet.py --data_dir ./datasets/300W_LP --filename_list ./datasets/300W_LP/300W_LP_Test_Split_1.txt --snapshot output/snapshots/ResNet34_epoch_'+str(i)+'.pkl  --dataset Pose_300W_LP --arch ResNet34'
    command = 'python test_hopenet.py --dataset BIWI --data_dir ./datasets/BIWI/test --filename_list ./datasets/BIWI/test/biwi_test.txt --snapshot output/snapshots/ResNet18_epoch_'+str(i)+'.pkl  --arch ResNet18'

    os.system(command)
    print("----------")