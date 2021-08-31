import os

for i in range(20,26):
    print("------------")
    print(f"model_no:{i}")
    #command = 'python test_hopenet.py --data_dir ./datasets/AFLW2000 --filename_list ./datasets/AFLW2000/files.txt --snapshot output/snapshots/_epoch_'+str(i)+'.pkl  --dataset AFLW2000 --arch Squeezenet_1_0'
    #command = 'python test_hopenet.py --data_dir ./datasets/300W_LP --filename_list ./datasets/300W_LP/300W_LP_Test_Split_1.txt --snapshot output/snapshots/_epoch_'+str(i)+'.pkl  --dataset Pose_300W_LP --arch Squeezenet_1_0'
    command = 'python test_hopenet.py --dataset BIWI --data_dir ./datasets/BIWI/test --filename_list ./datasets/BIWI/test/biwi_test.txt --snapshot output/snapshots/_epoch_'+str(i)+'.pkl  --arch Squeezenet_1_1'
  
    os.system(command)
    print("----------")
