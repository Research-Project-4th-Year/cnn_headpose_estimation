import os

for i in range(1,26):
    print("------------")
    print(f"model_no:{i}")
    command = 'python test_hopenet.py --data_dir ./datasets/AFLW2000 --filename_list ./datasets/AFLW2000/files.txt --snapshot output/snapshots/ResNet50_epoch_'+str(i)+'.pkl  --dataset AFLW2000 --arch ResNet50'
    os.system(command)
    print("----------")