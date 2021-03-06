import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

import datasets, hopenet, hopelessnet, utils, seresnet50, densenet201

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu',
        dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument('--data_dir', 
        dest='data_dir', help='Directory path for data.',
        default='datasets/AFLW2000', type=str)
    parser.add_argument('--filename_list', 
        dest='filename_list', 
        help='Path to text file containing relative paths for every example.',
        default='datasets/AFLW2000/files.txt', type=str)
    parser.add_argument('--snapshot', 
        dest='snapshot', help='Name of model snapshot.', 
        default='', type=str)
    parser.add_argument('--batch_size', 
        dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument('--save_viz', 
        dest='save_viz', help='Save images with pose cube.',
        default=False, type=bool)
    parser.add_argument('--dataset', 
        dest='dataset', help='Dataset type.', 
        default='AFLW2000', type=str)
    parser.add_argument(
        '--arch', dest='arch', 
        help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], '
            'ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot

    # Base network structure
    if args.arch == 'ResNet18':
        model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
    elif args.arch == 'ResNet34':
        model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [3,4,6,3], 66)
    elif args.arch == 'ResNet101':
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
    elif args.arch == 'ResNet152':
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
    elif args.arch == 'Squeezenet_1_0':
        model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
    elif args.arch == 'Squeezenet_1_1':
        model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
    elif args.arch == 'MobileNetV2':
        model = hopelessnet.Hopeless_MobileNetV2(66, 1.0)
    elif args.arch == 'SEResNet50':
        model = seresnet50.se_resnet50(num_classes=66)
        pre_url = 'https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl'
    elif args.arch == 'DenseNet201':
        model = densenet201.DenseNet_HopeNet(32, (6, 12, 24, 16), 64,66)
        pre_url = 'https://download.pytorch.org/models/densenet201-c1103571.pth'
    else:
        if args.arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000_ds':
        pose_dataset = datasets.AFLW2000_ds(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWINEW(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(
            args.data_dir, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        num_workers=2)

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    l1loss = torch.nn.L1Loss()

    for i, (images, labels, cont_labels, name) in enumerate(test_loader):
        images = Variable(images).cuda(gpu)
        total += cont_labels.size(0)

        label_yaw = cont_labels[:,0].float()
        label_pitch = cont_labels[:,1].float()
        label_roll = cont_labels[:,2].float()

        if args.arch == 'ResNet50' or args.arch == 'ResNet34' or args.arch == 'ResNet18' or args.arch == 'SEResNet50':
            x1, x2, x3, x4, x5, x6, yaw, pitch, roll = model(images)
            #yaw, pitch, roll = model(images)
        elif args.arch == 'Squeezenet_1_0' or args.arch == 'Squeezenet_1_1' or args.arch == 'DenseNet201' or args.arch == 'MobileNetV2':
            x1, yaw, pitch, roll = model(images)

        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = \
            torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = \
            torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = \
            torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        # Mean absolute error
        yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
        pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
        roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

        # Save first image in batch with pose cube or axis.
        if args.save_viz:
            name = name[0]
            if args.dataset == 'BIWI':
                cv2_img = cv2.imread(
                    os.path.join(args.data_dir, name + '_rgb.png'))
            else:
                cv2_img = cv2.imread(
                    os.path.join(args.data_dir, name + '.jpg'))
            if args.batch_size == 1:
                error_string = 'y %.2f, p %.2f, r %.2f' % (
                    torch.sum(torch.abs(yaw_predicted - label_yaw)), 
                    torch.sum(torch.abs(pitch_predicted - label_pitch)), 
                    torch.sum(torch.abs(roll_predicted - label_roll)))
                cv2.putText(
                    cv2_img, 
                    error_string, 
                    (30, cv2_img.shape[0]- 30), 
                    fontFace=1, 
                    fontScale=1, 
                    color=(0,0,255), 
                    thickness=2)
            
            # utils.plot_pose_cube(
            #   cv2_img, yaw_predicted[0], 
            #   pitch_predicted[0], 
            #   roll_predicted[0], 
            #   size=100)
            
            utils.draw_axis(
                cv2_img, yaw_predicted[0], 
                pitch_predicted[0], 
                roll_predicted[0], 
                tdx = 200, tdy= 200, size=100)
            cv2.imwrite(
                os.path.join('output/images', name + '.jpg'), cv2_img)

    print('Test error in degrees of the model on the ' + str(total) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
        yaw_error / total, pitch_error / total, roll_error / total,
        (yaw_error + pitch_error + roll_error) / (total * 3)))
