# # # ---> Import Packages <---

import os

import logging.config
from collections import defaultdict

import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from configuration import config 
from utils.data_loader import get_train_datalist, get_test_datalist
from utils.data_loader import get_statistics
from utils.method_manager import select_method
from utils.losses import SupConLoss, CeCLLoss

import warnings
warnings.filterwarnings('ignore')


def main():
    
    # # ---> 'args' (defines experimental settings) <---
    args = config.base_parser()
    
    # # PATH to save the results
    save_path = f"{args.dataset}/{args.mode}_online_disjoint_msz{args.memory_size}_rnd{args.seed}_mbsz100"
    # save_path = f"{args.dataset}/{args.mode}_online_no-memory_rnd{args.seed}"

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"logs/{args.dataset}", exist_ok=True)
    fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    writer = SummaryWriter("tensorboard")


    # # Set GPU device if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")
    
    logger.info(f"Arguments: {args}")
    
    # # --- Get dataset statistics ---
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)


    # # --- Dataset transformation ---
    train_transform = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        # transforms.RandomResizedCrop(size=(inp_size, inp_size), scale=(0.2, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # # # Optional - ColorJitter and GrayScale:
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        # ], p=0.8),  # Apply ColorJitter with 80% probability
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])
    logger.info(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ]
    )
    logger.info(f"Using test-transforms {test_transform}")


    logger.info(f"[1] Select a CIL method ({args.mode})")


    # # --- Define the LOSS function ---    
    if args.mode == "clai" or args.mode == "scr" or args.mode == "clai2":
        criterion = SupConLoss(temperature=0.1)
    elif args.mode == "er_las":
        criterion = nn.CrossEntropyLoss()
    elif args.mode == "cecr":
        criterion = CeCLLoss()
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")


    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )

    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    
    
    task_records = defaultdict(list)
    samples_cnt = 0
    pred_bias = []

    # # --- Incrementally 'train' and 'test' on each 'task' ---
    for cur_iter in range(args.n_tasks):

        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        task_acc = 0.0
        eval_dict = dict()


        # # get datalist
        cur_train_datalist = get_train_datalist(args, cur_iter)
        cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)

        # # Equally split the 'test' set to get 'test_datalist' and 'valid_datalist'
        # partition = len(test_datalist) // 2
        # test_datalist = test_datalist[partition:]
        # valid_datalist = test_datalist[:partition]

        logger.info("[2-2] Set environment for the current task")
        method.set_current_dataset(cur_train_datalist, cur_test_datalist)
        # # Increment known class for current task iteration.
        method.before_task(cur_train_datalist, cur_iter, args.init_model, args.init_opt)

        # # # The way to handle streamed samles
        # logger.info(f"[2-3] Start to train under {args.stream_env}")

        if args.stream_env == "offline" or args.mode == "joint" or args.mode == "gdumb":
            # if args.mode == "clib":
            #     # method.online_before_task(cur_iter)
            #     for i, data in enumerate(cur_train_datalist):
            #         samples_cnt += 1
            #         method.online_step(data, samples_cnt, args.n_worker)
            #     # method.online_after_task(cur_iter)
            # # Offline Train
            # else:
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )
            if args.mode == "joint":
                logger.info(f"joint accuracy: {task_acc}")

        # if args.mode == "rm":
        #     method.update_memory(cur_iter)

        #     logger.info("### Train using Memory data only for RM ###")
        #     method.set_current_dataset([], cur_test_datalist)
        #     task_acc, eval_dict = method.train(
        #         cur_iter=cur_iter,
        #         n_epoch=args.n_epoch,
        #         batch_size=args.batchsize,
        #         n_worker=args.n_worker,
        #     )
        
        # # logger.info("[2-4] Update the information for the current task")
        method.after_task(cur_iter)
        task_records["task_acc"].append(task_acc)
        # if cur_iter > 0:
        #     pred_bias.append(eval_dict["old_to_new_misclass_pct"])
        
        # task_records['cls_acc'][k][j] = break down j-class accuracy from 'task_acc'
        # Convert cls_acc to flat list if it's a dict
        if isinstance(eval_dict["cls_acc"], dict):
            cls_acc_list = [eval_dict["cls_acc"][i] for i in sorted(eval_dict["cls_acc"].keys())]
        else:
            cls_acc_list = eval_dict["cls_acc"]

        task_records["cls_acc"].append(cls_acc_list)
        task_records["cls_acc"].append(eval_dict["cls_acc"])

        # Notify to NSML
        logger.info("[2-5] Report task result")
        writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)

        logger.info(f"task records (class accuracies) {task_records['cls_acc']}")
        logger.info(f"task records {task_records['task_acc']}")

    # np.save(f"results/{save_path}.npy", task_records["task_acc"])

    # # # # ---> Accuracy (A) <---
    # A_avg = np.mean(task_records["task_acc"])
    # A_last = task_records["task_acc"][args.n_tasks - 1]
    
    
    # # # # ---> Forgetting (F) <---
    # acc_arr = np.array(task_records["cls_acc"])
    
    # # # cls_acc = (k, j), acc for j at k
    # cls_acc = acc_arr.reshape(-1, args.n_cls_a_task).mean(1).reshape(args.n_tasks, -1)
    # # # cls_acc = acc_arr.reshape(args.n_tasks, 100).mean(axis=0)
    # # cls_acc = acc_arr.mean(axis=1)
    
    # for k in range(args.n_tasks):
    #     forget_k = []
    #     for j in range(args.n_tasks):
    #         if j < k:
    #             forget_k.append(cls_acc[:k, j].max() - cls_acc[k, j])
    #             # forget_k.append(cls_acc[:k].max() - cls_acc[k])
    #         else:
    #             forget_k.append(None)
    #     task_records["forget"].append(forget_k)
    # F_last = np.mean(task_records["forget"][-1][:-1])

    # acc_arr = np.array(task_records["cls_acc"])

    # # No reshaping using n_cls_a_task, directly calculate task-level mean
    # # cls_acc = acc_arr.mean(axis=1) gives average accuracy over classes for each task.
    # cls_acc = acc_arr.mean(axis=1)  # Now each task will have a mean accuracy

    # # Calculate forgetting measure per task
    # for k in range(args.n_tasks):
    #     forget_k = []
    #     for j in range(args.n_tasks):
    #         if j < k:
    #             # Compare max accuracy up to task k for each class j with current task's accuracy
    #             forget_k.append(cls_acc[:k].max() - cls_acc[k])
    #         else:
    #             forget_k.append(None)  # No forgetting for tasks not yet trained
    #     task_records["forget"].append(forget_k)

    # # Calculate the forgetting rate for the last task
    # F_last = np.mean([f for f in task_records["forget"][-1][:-1] if f is not None])

    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]

    # Forgetting (F)
    acc_arr = np.array(task_records["cls_acc"])
    # cls_acc = (k, j), acc for j at k
    cls_acc = acc_arr.reshape(-1, args.n_cls_a_task).mean(1).reshape(args.n_tasks, -1)
    for k in range(args.n_tasks):
        forget_k = []
        for j in range(args.n_tasks):
            if j < k:
                forget_k.append(cls_acc[:k, j].max() - cls_acc[k, j])
            else:
                forget_k.append(None)
        task_records["forget"].append(forget_k)
    F_last = np.mean(task_records["forget"][-1][:-1])


    # # # ---> Intrasigence (I) <---
    I_last = args.joint_acc - A_last

    logger.info(f"======== Summary =======")
    logger.info(f"A_last {A_last} | A_avg {A_avg} | F_last {F_last} | I_last {I_last}")
    logger.info(f"Average Prediction Bias over all tasks: {np.mean(pred_bias)}")


if __name__ == "__main__":
    main()
