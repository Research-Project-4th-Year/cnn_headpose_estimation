import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


import datasets, hopenet, hopelessnet, seresnet50, densenet201, utils, at_loss
import torch.utils.model_zoo as model_zoo
import time
start_time = time.time()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', 
        help='Maximum number of training epochs.',
        default=50, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.000001, type=float)
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.', 
        default='Pose_300W_LP', type=str)
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)
    parser.add_argument(
        '--filename_list', dest='filename_list', 
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str)
    parser.add_argument(
        '--filename_list_train', dest='filename_list_train', 
        help='Path to text file containing relative paths for every example.',
        default='datasets/BIWI/biwi_train.txt', type=str)
    parser.add_argument(
        '--filename_list_test', dest='filename_list_test', 
        help='Path to text file containing relative paths for every example.',
        default='datasets/BIWI/biwi_test.txt', type=str)
    parser.add_argument(
        '--output_string', dest='output_string', 
        help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--arch', dest='arch', 
        help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], '
            'ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
    parser.add_argument('--p', type=float, default=2.0, help='power for AT')


    args = parser.parse_args()
    return args

def get_ignored_params(model, arch):
    # Generator function that yields ignored params.
    if arch.find('ResNet') >= 0 or arch.find('SEResNet50') >= 0:
        b = [model.conv1, model.bn1, model.fc_finetune]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0 or arch.find('DenseNet201') >= 0:
        b = [model.features[0]]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model, arch):
    # Generator function that yields params that will be optimized.
    if arch.find('ResNet') >= 0 or arch.find('SEResNet50') >= 0:
        b = [model.layer1, model.layer2, model.layer3, model.layer4]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0 or arch.find('DenseNet201') >= 0:
        b = [model.features[1:]]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model, arch):
    # Generator function that yields fc layer params.
    if arch.find('ResNet') >= 0 or arch.find('SEResNet50') >= 0:
        b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0 or arch.find('DenseNet201') >= 0:
        b = [
            model.classifier_yaw, 
            model.classifier_pitch, 
            model.classifier_roll
        ]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    try:
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        print('Ignoring "' + str(e) + '"')


def write_training_result(train_losses, total_errors, arch):
    num_epochs = len(train_losses)

    file_name = './output/train_test_result/train_result_'+arch+'.txt'
    with open(file_name,'w') as f:
        for i in range(0,num_epochs):
            result = str(round(train_losses[i], 4)) + ',' \
                   + str(round(total_errors[i], 4)) + '\n'
            f.write(result)

def test(val_loader, model, args):
    
    model.eval()
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    with torch.no_grad():
        for i, (images, labels, cont_labels, name) in enumerate(val_loader):
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)    

            # Forward pass
            if args.arch == 'ResNet50' or args.arch == 'ResNet34' or args.arch == 'ResNet18' or args.arch == 'SEResNet50':
                x1, x2, x3, x4, x5, x6, yaw, pitch, roll = model(images)
            elif args.arch == 'Squeezenet_1_0' or args.arch == 'Squeezenet_1_1' or args.arch == 'DenseNet201' or args.arch == 'MobileNetV2':
                x1, yaw, pitch, roll = model(images)

            #-----------------------------------
            #Calavulate MAE 
            label_yaw_cpu = cont_labels[:,0].float()
            label_pitch_cpu = cont_labels[:,1].float()
            label_roll_cpu = cont_labels[:,2].float()

          
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
            yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw_cpu))
            pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch_cpu))
            roll_error += torch.sum(torch.abs(roll_predicted - label_roll_cpu))

        total_error = (yaw_error + pitch_error + roll_error)/(total *3)

        return yaw_error, pitch_error, roll_error, total_error.item(), total

def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    #########################Student architecture###############################
    if args.arch == 'ResNet18':
        student_model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif args.arch == 'ResNet34':
        student_model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [3,4,6,3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif args.arch == 'ResNet101':
        student_model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif args.arch == 'ResNet152':
        student_model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    elif args.arch == 'Squeezenet_1_0':
        student_model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
        pre_url = \
            'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth'
    elif args.arch == 'Squeezenet_1_1':
        student_model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
        pre_url = \
            'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'
    elif args.arch == 'MobileNetV2':
        student_model = hopelessnet.Hopeless_MobileNetV2(66, 1.0)
        pre_url = \
            'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
    elif args.arch == 'SEResNet50':
        student_model = seresnet50.se_resnet50(num_classes=66)
        pre_url = 'https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl'
    elif args.arch == 'DenseNet201':
        student_model = densenet201.DenseNet_HopeNet(32, (6, 12, 24, 16), 64,66)
        pre_url = 'https://download.pytorch.org/models/densenet201-c1103571.pth'
    else:
         print('Invalid value for architecture is passed!')
         exit()

    if args.snapshot == '':
        load_filtered_state_dict(student_model, model_zoo.load_url(pre_url))
    else:
        saved_state_dict = torch.load(args.snapshot)
        student_model.load_state_dict(saved_state_dict)

    print(f"Student Netowrk Size: {count_parameters_in_MB(student_model)}MB")
    print("Student: ",student_model)

    #####################Teacher Network - ResNet50###############################
    teacher_model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    saved_state_dict = torch.load('output/snapshots/basic_models/ResNet50_Basic.pkl')
    teacher_model.load_state_dict(saved_state_dict)

    for param in teacher_model.parameters():
        param.requires_grad = False
    print(f"Teacher Netowrk Size: {count_parameters_in_MB(teacher_model)}MB")
    print("Teacher:",teacher_model)



    print('Loading data.')
    transformations = transforms.Compose([transforms.Resize(240),
        transforms.RandomCrop(224), transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
            )
        ])

   
    if args.dataset == 'BIWI':
        pose_dataset_train = datasets.BIWINEW(
            args.data_dir, args.filename_list_train, transformations)
        pose_dataset_test = datasets.BIWINEW(
            args.data_dir, args.filename_list_test, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    
    print("Train Set:"+ str(len(pose_dataset_train)))
    print("Test Set :"+ str(len(pose_dataset_test)))

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset_test,
        batch_size=1,
        num_workers=2)
    
    student_model.cuda(gpu)
    teacher_model.cuda(gpu)

    teacher_model.eval()

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    #define kd loss function
    kd_criterion = at_loss.AT(args.p)
  

    softmax = nn.Softmax(dim=1).cuda(gpu)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([
        {'params': get_ignored_params(student_model, args.arch), 'lr': 0},
        {'params': get_non_ignored_params(student_model, args.arch), 'lr': args.lr},
        {'params': get_fc_params(student_model, args.arch), 'lr': args.lr * 5}
        ], lr = args.lr)

    

    # Log variables
    train_losses = []
    test_errors = []
    total_train_loss = .0
    loss_yaw_total = .0
    loss_pitch_total = .0
    loss_roll_total = .0
    total_train_labels = 0
    minimum_error = 10000.0 #Minimum MAE
    counter = 0

    print('Ready to train network.')
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)
            total_train_labels += cont_labels.size(0)
    
            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            if args.arch == 'ResNet50' or args.arch == 'ResNet34' or args.arch == 'ResNet18' or args.arch == 'SEResNet50':
                x1_s, x2_s, x3_s, x4_s, x5_s, x6_s, yaw_s, pitch_s, roll_s = student_model(images)
            elif args.arch == 'Squeezenet_1_0' or args.arch == 'Squeezenet_1_1' or args.arch == 'DenseNet201' or args.arch == 'MobileNetV2':
                x1_s, yaw_s, pitch_s, roll_s = student_model(images)

            x1_t, x2_t, x3_t, x4_t, x5_t, x6_t, yaw_t, pitch_t, roll_t = teacher_model(images)

            print("Teacher1 :",x1_t[1].size())
            print("Student1 :",x1_s[1].size())
            print("Teacher2 :",x2_t[1].size())
            print("Student2 :",x2_s[1].size())
            print("Teacher3 :",x3_t[1].size())
            print("Student3 :",x3_s[1].size())
            print("Teacher4 :",x4_t[1].size())
            print("Student5 :",x4_s[1].size())
            exit()

            # Cross entropy loss
            loss_yaw = criterion(yaw_s, label_yaw) 
            loss_pitch = criterion(pitch_s, label_pitch) 
            loss_roll = criterion(roll_s, label_roll)

            # student loss with AT
            kd_loss = (kd_criterion(x1_s[1], x1_t[1].detach()) +
					   kd_criterion(x2_s[1], x2_t[1].detach()) +
					   kd_criterion(x3_s[1], x3_t[1].detach()) +
                       kd_criterion(x4_s[1], x4_t[1].detach())) / 4.0 * args.lambda_kd

          
            loss_yaw +=kd_loss
            loss_pitch +=kd_loss
            loss_roll +=kd_loss
                

            # MSE loss
            yaw_predicted = softmax(yaw_s)
            pitch_predicted = softmax(pitch_s)
            roll_predicted = softmax(roll_s)


            yaw_predicted = \
                torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = \
                torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = \
                torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99


            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)


            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            #Total training loss : Log purpose 
            loss_yaw_total += loss_yaw.item()
            loss_pitch_total += loss_pitch.item()
            loss_roll_total += loss_roll.item()

            # Compute gradient and do SGD step
            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = \
                [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: '
                    'Yaw %.4f, Pitch %.4f, Roll %.4f'%(
                        epoch+1, 
                        num_epochs, 
                        i+1, 
                        len(pose_dataset_train)//batch_size, 
                        loss_yaw.item(), 
                        loss_pitch.item(), 
                        loss_roll.item()
                    )
                )


       
        total_train_loss = (loss_yaw_total + loss_pitch_total + loss_roll_total)/(total_train_labels *3)
        train_losses.append(total_train_loss)

        #Measure the MAE 
        yaw_error, pitch_error, roll_error, total_error, total = test(test_loader, student_model, args)
        test_errors.append(total_error)

        print('Validation error in degrees of the model on the ' + str(total) +
                    ' validation images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
                    yaw_error / total, pitch_error / total, roll_error / total,
                    (yaw_error + pitch_error + roll_error) / (total * 3)))
        
        #Clear data to track next epoch
        loss_yaw_total=.0
        loss_pitch_total = .0
        loss_roll_total = .0
        total_train_labels = 0


        #Save the best model..
        if total_error < minimum_error:
                print('Taking snapshot...',
                    torch.save(student_model.state_dict(),
                    'output/snapshots/' + args.output_string + 
                    str(args.arch)+'_Basic'+ '.pkl')
                )

                minimum_error = total_error


        student_model.train()

    write_training_result(train_losses, test_errors, args.arch)


print("--- %s Train Time (seconds) ---" % (time.time() - start_time))