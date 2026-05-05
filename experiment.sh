#!/bin/bash

# CIL CONFIG
MODE="pcr" # joint, finetune, ewc, er, rm, er_obc, er_las, clai
# "default": If you want to use the default memory management method.
MEM_MANAGE="default" 
### finetune & ewc ---> reservoir (if EM is used) | er ---> reservoir | rm ---> uncertainty
### For other methods, reservoir is implememted explicitly
RND_SEED=1 # FIXED
DATASET="cifar100" # cifar10, cifar100, miniimagenet, core50
STREAM="offline" # offline, online (NOT implemented yet)
EXP="disjoint" # disjoint, (blurry10, blurry30 - NOT implemented yet)
MEM_SIZE=2000 # cifar10: k={0.2k, 0.5k, 1k}, cifar100: k={0.5k, 1k, 2k}, mini-imagenet: k={0.5k, 1k, 2k}, core50: k={1k, 2k}
SEED_VALUE=0 # Variable = {1, 2, 3}
TRANS="" # multiple choices: cutmix, cutout, randaug, autoaug

N_WORKER=4
JOINT_ACC=0.0 # training all the tasks at once.
### FINISH CIL CONFIG ###

# RM
UNCERT_METRIC="vr"
PRETRAIN="" INIT_MODEL="" INIT_OPT=""

# iCaRL
FEAT_SIZE=512

# BiC
distilling="--distilling" # Normal BiC. If you do not want to use distilling loss, then "".


if [ -d "tensorboard" ]; then
    rm -rf tensorboard
    echo "Remove the tensorboard dir"
fi

if [ "$DATASET" == "mnist" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="mlp400"
    N_EPOCH=1; BATCHSIZE=100; LR=0.001 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    # elif [[ "$EXP" == *"blurry"* ]]; then
    #     N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "fmnist" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="mlp400"
    N_EPOCH=1; BATCHSIZE=10; LR=0.1 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar10" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="resnet18"
    N_EPOCH=1; BATCHSIZE=10; LR=0.1 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    # elif [[ "$EXP" == *"blurry"* ]]; then
    #     N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar100" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=100 TOPK=1
    MODEL_NAME="resnet18"
    N_EPOCH=1; BATCHSIZE=10; LR=0.1 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=100 N_CLS_A_TASK=55 N_TASKS=10
    else
        N_INIT_CLS=100 N_CLS_A_TASK=10 N_TASKS=10
    fi

elif [ "$DATASET" == "miniimagenet" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=100 TOPK=1
    MODEL_NAME="resnet34"
    N_EPOCH=1; BATCHSIZE=10; LR=0.1 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=1
    # elif [[ "$EXP" == *"blurry"* ]]; then
    #     N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    else
        N_INIT_CLS=100 N_CLS_A_TASK=10 N_TASKS=10
    fi
elif [ "$DATASET" == "tinyimagenet" ]; then
    TOTAL=100000 N_VAL=0 N_CLASS=200 TOPK=1
    MODEL_NAME="resnet34"
    N_EPOCH=1; BATCHSIZE=10; LR=0.1 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=200 N_CLS_A_TASK=200 N_TASKS=1
    # elif [[ "$EXP" == *"blurry"* ]]; then
    #     N_INIT_CLS=200 N_CLS_A_TASK=20 N_TASKS=10
    else
        N_INIT_CLS=200 N_CLS_A_TASK=20 N_TASKS=10
    fi
elif [ "$DATASET" == "core50" ]; then
    TOTAL=100000 N_VAL=0 N_CLASS=50 TOPK=1
    MODEL_NAME="resnet34"
    N_EPOCH=1; BATCHSIZE=10; LR=0.1 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=50 N_CLS_A_TASK=5 N_TASKS=1
    # elif [[ "$EXP" == *"blurry"* ]]; then
    #     N_INIT_CLS=50 N_CLS_A_TASK=50 N_TASKS=11
    else
        N_INIT_CLS=50 N_CLS_A_TASK=5 N_TASKS=10
    fi

elif [ "$DATASET" == "imagenet" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=1000 TOPK=5
    MODEL_NAME="resnet34"
    N_EPOCH=2; BATCHSIZE=256; LR=0.05 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    else
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=10
    fi
else
    echo "Undefined setting"
    exit 1
fi

python main.py --mode $MODE --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET \
--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED --seed $SEED_VALUE \
--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--n_worker $N_WORKER --n_epoch $N_EPOCH \
--memory_size $MEM_SIZE --transform $TRANS --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC