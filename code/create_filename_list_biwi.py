import os, sys, argparse
import numpy as np
import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Create filenames list txt file from datasets root dir.'
        ' For head pose analysis.')
    parser.add_argument('--root_dir = ', 
        dest='root_dir', 
        help='root directory of the datasets files', 
        default='./datasets/BIWI_2/BIWI_Preprocess/train', 
        type=str)
    parser.add_argument('--filename', 
        dest='filename', 
        help='Output filename.',
        default='biwi_train.txt', 
        type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    os.chdir(args.root_dir)

    file_counter = 0
    rej_counter = 0
    outfile = open(args.filename, 'w')

    for root, dirs, files in os.walk('.'): 
        for f in files: 
            if f[-4:] == '.jpg': 
               
                posefile_file_name = os.path.join(root, f.replace('.jpg', '.txt'))
                posefile_file_name = open(posefile_file_name, 'r')
                pose_value = posefile_file_name.read()
                pose_value = pose_value[:-1]
                
                
                yaw, pitch, roll = pose_value.split(' ')
                yaw, pitch, roll = float(yaw), float(pitch), float(roll)

                if abs(pitch) <= 99 and abs(yaw) <= 99 and abs(roll) <= 99:
                    if file_counter > 0:
                        outfile.write('\n')
                    outfile.write(root + '/' + f[:-4])
                    file_counter += 1
                else:
                    rej_counter += 1
    outfile.close()
    print(f'{file_counter} files listed! {rej_counter} files had out-of-range'
        f' values and kept out of the list!')

