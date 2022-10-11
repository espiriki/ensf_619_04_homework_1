#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist
from utils import get_dataset, average_weights, exp_details
from torchsummary import summary

def print_summary(model):

    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for _, (image, _) in enumerate(trainloader):
        summary(model, image)
        break


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)

            print_summary(global_model)

            pytorch_total_params = sum(p.numel() for p in global_model.parameters())

            print("Total Parameters CNN: {}".format(pytorch_total_params))

            # According to the paper, the number of parameters in the CNN model is 1663370
            assert pytorch_total_params == 1663370

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x

        # According to the paper, the number of neurons in the hidden
        # layer is 200
        global_model = MLP(dim_in=len_in, dim_hidden=200,
                            dim_out=args.num_classes)

        print_summary(global_model)

        pytorch_total_params = sum(p.numel() for p in global_model.parameters())

        print("Total Parameters 2NN: {}".format(pytorch_total_params))

        # According to the paper, the number of parameters in the 2NN model is 199210
        assert pytorch_total_params == 199210

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy, test_accuracy = [], [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0

    print(args)

    epoch_count = 0
    for epoch in range(args.epochs):

        # if epoch_count%38==0 and epoch_count > 0:
        #     args.lr = args.lr * 1.0/(pow(10.0,1.0/6.0))

        # init local weights and loss
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        # sample a fraction of users (with args frac)
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        print("Sampling {} devices with ids:".format(m))
        print(idxs_users)

        # for each sampled device, run local training
        print("Local training started with LR: {}".format(args.lr))
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))


        print("Local training finished!\n")
        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # calculate global loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        print("Evaluating global model...")
        # calculate test set accuracy
        test_acc, _ = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)

        print(f'Avg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Test Set Accuracy: {:.2f}% \n'.format(100*test_acc))

        epoch_count = epoch_count + 1

        if args.model == 'mlp' and test_acc > 0.92:
            break

        if args.model == 'cnn' and test_acc > 0.94:
            break

    print(f' \n Results after {epoch_count} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and test_accuracy:
    file_name = 'save/objects/{}_{}_epochs[{}]_frac_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, epoch_count, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, test_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('save/fed_{}_{}_epochs[{}]_frac_C[{}]_iid[{}]_E[{}]_B[{}]_LR[{}]_loss.png'.
                format(args.dataset, args.model, epoch_count, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Test Accuracy vs Communication rounds')
    plt.plot(range(len(test_accuracy)), test_accuracy, color='k')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('save/fed_{}_{}_epochs[{}]_frac_C[{}]_iid[{}]_E[{}]_B[{}]_LR[{}]_acc.png'.
                format(args.dataset, args.model, epoch_count, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr))
