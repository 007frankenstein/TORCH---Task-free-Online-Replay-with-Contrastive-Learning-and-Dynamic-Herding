import logging

from methods.finetune import Finetune
from methods.joint import Joint

from methods.regularization import EWC #, RWalk
from methods.exp_replay import ER

# from methods.icarl import ICaRL
# from methods.gdumb import GDumb
from methods.bic import BiasCorrection

from methods.derpp import DERPP
from methods.rainbow_memory import RM
from methods.er_obc import ERobc
from methods.oser import OSER
from methods.er_las import ER_LAS

from methods.clai import CLAI
from methods.clai2 import CLAI2

from methods.scr import SCR
from methods.pcr import PCR

from methods.cecr import CeCR

logger = logging.getLogger()


def select_method(args, criterion, device, train_transform, test_transform, n_classes):
    kwargs = vars(args)
    if args.mode == "finetune":
        method = Finetune(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "joint":
        method = Joint(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )

    elif args.mode == "ewc":
        method = EWC(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    # elif args.mode == "rwalk":
    #     method = RWalk(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    elif args.mode == "er":
        method = ER(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )

    # elif args.mode == "icarl":
    #     method = ICaRL(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    # elif args.mode == "gdumb":
    #     method = GDumb(
    #         criterion=criterion,
    #         device=device,
    #         train_transform=train_transform,
    #         test_transform=test_transform,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    elif args.mode == "bic":
        method = BiasCorrection(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )

    elif args.mode == "derpp":
        method = DERPP(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "rm":
        method = RM(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_obc":
        method = ERobc(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )    
    elif args.mode == "oser":
        method = OSER(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_las":
        method = ER_LAS(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )

    elif args.mode == "clai":
        method = CLAI(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )

    elif args.mode == "clai2":
        method = CLAI2(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    
    elif args.mode == "scr":
        method = SCR(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "pcr":
        method = PCR(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )

    elif args.mode == "cecr":
        method = CeCR(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )

    else:
        raise NotImplementedError("Choose the args.mode in [finetune, gdumb]")

    logger.info("CIL Scenario: ")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_init_cls: {args.n_init_cls}")
    print(f"n_cls_a_task: {args.n_cls_a_task}")
    print(f"total cls: {args.n_tasks * args.n_cls_a_task}")

    return method
