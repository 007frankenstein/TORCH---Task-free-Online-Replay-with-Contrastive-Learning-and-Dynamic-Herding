
### When we make a new one, we should inherit the Finetune class. ###
import logging
import os
import random

import PIL
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from randaugment.randaugment import RandAugment

from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from utils.data_loader import ImageDataset
from utils.data_loader import cutmix_data
from utils.train_utils import select_model, select_optimizer
from utils.labelSmoothing_loss import LabelSmoothingCrossEntropy

from models.resnet import SupConResNet, Reduced_ResNet18, CECRModel
# from models.resnet18 import SupConResNet, Reduced_ResNet18
# from sklearn.cluster import KMeans
from tsne import extract_features
from utils.utils import l2_normalize, extract_memory_features, plot_tsne_subset_classes



logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class ICaRLNet(nn.Module):
    def __init__(self, model, feature_size, n_class):
        super().__init__()
        self.model = model
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.linear = nn.Linear(feature_size, n_class, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.linear(x)
        return x


# # # ---> Base Class <---
class Finetune:
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        self.num_learned_class = 0
        self.num_learning_class = kwargs["n_init_cls"]
        self.n_classes = n_classes
        self.learned_classes = []
        self.class_mean = [None] * n_classes
        self.exposed_classes = []
        self.seen = 0
        self.topk = kwargs["topk"]

        self.device = device
        self.criterion = criterion
        self.dataset = kwargs["dataset"]
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        self.lr = kwargs["lr"]
        self.feature_size = kwargs["feature_size"]

        self.train_transform = train_transform
        self.cutmix = "cutmix" in kwargs["transforms"]
        self.test_transform = test_transform

        self.prev_streamed_list = []
        self.streamed_list = []
        self.test_list = []
        self.memory_list = []
        self.memory_size = kwargs["memory_size"]
        self.mem_manage = kwargs["mem_manage"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "reservoir"
        random.seed(kwargs["seed"])
        torch.manual_seed(kwargs["seed"])
        np.random.seed(kwargs["seed"])

        self.task_classes = defaultdict(list)

        if self.dataset == "miniimagenet" or self.dataset == "core50":
            dim_in = 640
        # elif self.dataset == "core50":
        #     dim_in = 2560
        else:
            dim_in = 160
        
        if kwargs["mode"] == "clai" or kwargs["mode"] == "pcr":
            self.model = SupConResNet(num_classes=kwargs["n_init_cls"], dim_in=dim_in, head="mlp")
        elif kwargs["mode"] == "cecr":
            self.model = Reduced_ResNet18(kwargs["n_init_cls"])
        else:
            self.model = Reduced_ResNet18(kwargs["n_init_cls"])
            if self.dataset == "miniimagenet" or self.dataset == "core50":
                self.model.linear = nn.Linear(dim_in, kwargs["n_init_cls"], bias=True)
      
        self.criterion = self.criterion.to(self.device)
    
        if kwargs["mode"] == "clai":
            self.classifier = nn.Linear(dim_in, kwargs["n_init_cls"])
            self.classifier = self.classifier.to(self.device)

            self.classifier_criterion = nn.CrossEntropyLoss(reduction="mean") # For ER
            self.classifier_criterion = self.classifier_criterion.to(self.device)

            self.optimizer_fc, self.scheduler = select_optimizer(
                self.opt_name, self.lr, list(self.classifier.parameters()), self.sched_name
            )

        if kwargs["mode"] == "er_obc":
            self.surrogate_classifier = nn.Linear(dim_in, kwargs["n_init_cls"])
            self.surrogate_classifier = self.surrogate_classifier.to(self.device)

            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, list(self.model.parameters())+list(self.surrogate_classifier.parameters()), self.sched_name
                )
        
            self.classifier = nn.Linear(dim_in, kwargs["n_init_cls"])      
            self.classifier = self.classifier.to(self.device)

            self.classifier_criterion = LabelSmoothingCrossEntropy() # For ERobc
            self.classifier_criterion = self.classifier_criterion.to(self.device)

            self.optimizer_fc, self.scheduler = select_optimizer(
                self.opt_name, self.lr, list(self.classifier.parameters()), self.sched_name
            )  
        else:
            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, list(self.model.parameters()), self.sched_name
                )     
                    
        self.already_mem_update = False

        self.mode = kwargs["mode"]

        self.uncert_metric = kwargs["uncert_metric"]

        self.replay_list = [] # Replay Memory

        self.seen_classes = []

        self.cls_mean = [np.zeros(dim_in)] * kwargs["n_init_cls"] # class mean vector
        
        self.cls_counter = [0 for element in range(kwargs["n_init_cls"])]

        self.prototypes = {}

        self.cecr_memory = {}

        self.seen_labels = set() 

        self.class_counts = {i: 0 for i in range(kwargs["n_init_cls"])}

        # self.PATH = "./saved_models/last.pth" #  Path to Saved model 


    def set_current_dataset(self, train_datalist, test_datalist):
        random.shuffle(train_datalist)
        self.prev_streamed_list = self.streamed_list
        self.streamed_list = train_datalist
        self.test_list = test_datalist
        

    def before_task(self, datalist, cur_iter, init_model=False, init_opt=True):
        logger.info("Apply before_task")
        incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()
        self.exposed_classes = list(set(self.learned_classes + incoming_classes))
        self.num_learning_class = max(
            len(self.exposed_classes), self.num_learning_class
        )

        if self.mem_manage == "prototype":
            self.model.linear = nn.Linear(self.model.linear.in_features, self.feature_size)
            self.feature_extractor = self.model
            self.model = ICaRLNet(
                self.feature_extractor, self.feature_size, self.num_learning_class
            )

        # in_features = self.model.in_features
        # out_features = self.model.out_features
        # # To care the case of decreasing head
        # new_out_features = max(out_features, self.num_learning_class)

        # if init_model:
        #     # init model parameters in every iteration
        #     logger.info("Reset model parameters")
        #     self.model = select_model(self.model_name, self.dataset, new_out_features)
        # else:
        #     self.model.fc = nn.Linear(in_features, new_out_features)
        
        self.params = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }  # For regularzation methods

        self.model = self.model.to(self.device)

        if init_opt:
            # # reinitialize the optimizer and scheduler
            logger.info("Reset the optimizer and scheduler states")
            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, self.model, self.sched_name
            )

        # logger.info(f"Increasing the head of fc {out_features} -> {new_out_features}")

        self.already_mem_update = False
        

    def after_task(self, cur_iter):
        logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class
        # self.update_memory(cur_iter)

        # if cur_iter == 4:
        #     # class_to_task = build_class_to_task(self.task_classes)
        #     model = torch.load(f"./er_model.pt")
        #     m_dataset = ImageDataset(
        #         pd.DataFrame(self.memory_list),
        #         dataset=self.dataset,
        #         transform=self.test_transform,
        #     )
            
        #     m_loader = DataLoader(m_dataset,
        #         shuffle=False, #important
        #         batch_size=100,
        #         num_workers=0, 
        #     )
        #     features, class_labels = extract_memory_features(
        #         model,
        #         m_loader,
        #         self.device
        #     )

        #     features = l2_normalize(features)

        #     # Good qualitative plot
        #     # selected_classes = np.random.choice(np.arange(50), 10, replace=False).tolist()
        #     np.random.seed(0)
        #     selected_classes = np.random.choice(
        #         np.arange(50),
        #         10,
        #         replace=False
        #     ).tolist()

        #     plot_tsne_subset_classes(
        #         features,
        #         class_labels,
        #         selected_classes,
        #         # title="t-SNE of Replay Memory (Class-wise)",
        #         save_path="./figures/tsne_crossEntropy_0.5.png"
        #     ) 

        #     exit() 

        # if cur_iter == 4:
        #     model = torch.load("./er_model.pt")
        #     model.eval()

        #     m_dataset = ImageDataset(
        #         pd.DataFrame(self.test_list),
        #         dataset=self.dataset,
        #         transform=self.test_transform,
        #     )
            
        #     m_loader = DataLoader(m_dataset,
        #         shuffle=False, #important
        #         batch_size=256,
        #         num_workers=0, 
        #     )

        #     # Extract features and labels
        #     features, logits, labels = extract_features(model, m_loader, self.device)

        #     # Apply t-SNE to reduce the dimensionality of the features
        #     tsne = TSNE(n_components=2, 
        #                 n_iter=1000, 
        #                 perplexity=50,  # higher value for better global structure
        #                 learning_rate=500,  # adjusted learning rate
        #                 random_state=42)
        #     features_tsne = tsne.fit_transform(features)

        #     # Plot the t-SNE results
        #     plt.figure(figsize=(10, 8))
        #     sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=labels, palette='tab10', s=60, alpha=0.7)
        #     plt.title("(a) ER", fontsize=20)
        #     # plt.xlabel("t-SNE Component 1")
        #     # plt.ylabel("t-SNE Component 2")
        #     # plt.legend(title="Classes", fontsize=18, title_fontsize=16, loc="upper right")
            
        #     plt.savefig("tsne_ER_cifar10.png")
        #     plt.show()


    def update_memory(self, cur_iter, num_class=None):
        if num_class is None:
            num_class = self.num_learning_class

        if not self.already_mem_update:
            logger.info(f"Update memory over {num_class} classes by {self.mem_manage}")
            candidates = self.streamed_list + self.memory_list
            if len(candidates) <= self.memory_size:
                self.memory_list = candidates
                self.seen = len(candidates)
                logger.warning("Candidates < Memory size")
            else:
                if self.mem_manage == "random":
                    self.memory_list = self.rnd_sampling(candidates)
                elif self.mem_manage == "reservoir":
                        self.reservoir_sampling(self.streamed_list)
                elif self.mem_manage == "prototype":
                    self.memory_list = self.mean_feature_sampling(
                        exemplars=self.memory_list,
                        samples=self.streamed_list,
                        num_class=num_class,
                    )
                elif self.mem_manage == "uncertainty":
                    if cur_iter == 0:
                        self.memory_list = self.equal_class_sampling(
                            candidates, num_class
                        )
                    else:
                        self.memory_list = self.uncertainty_sampling(
                            candidates,
                            num_class=num_class,
                        )
                elif self.mem_manage == "no-mem":
                    self.memory_list = []
                else:
                    logger.error("Not implemented memory management")
                    raise NotImplementedError

            assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            if self.memory_list:
                logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")
            # # memory update happens only once per task iteratin.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")

    def get_dataloader(self, batch_size, n_worker, train_list, test_list):
        # # Loader
        train_loader = None
        test_loader = None
        if train_list is not None and len(train_list) > 0:
            train_dataset = ImageDataset(
                pd.DataFrame(train_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            # # drop last becasue of BatchNorm1D in IcarlNet
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=True,
            )

        if test_list is not None:
            test_dataset = ImageDataset(
                pd.DataFrame(test_list),
                dataset=self.dataset,
                transform=self.test_transform,
            )
            test_loader = DataLoader(
                test_dataset, shuffle=False, batch_size=batch_size, num_workers=n_worker
            )

        return train_loader, test_loader

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=1):

        train_list = self.streamed_list + self.memory_list
        random.shuffle(train_list)
        test_list = self.test_list
        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # # # ---> TRAIN <---
        best_acc = 0.0
        eval_dict = dict()
        for epoch in range(n_epoch):
            # https://github.com/drimpossible/GDumb/blob/master/src/main.py
            # # initialize for each task
            # if epoch <= 0:  # Warm start of 1 epoch
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = self.lr * 0.1
            # elif epoch == 1:  # Then set to maxlr
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = self.lr
            # else:  # Aand go!
            #     self.scheduler.step()

            train_loss, train_acc = self._train(
                train_loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                epoch=epoch,
                total_epochs=n_epoch,
                n_passes=n_passes,
            )

            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            best_acc = max(best_acc, eval_dict["avg_acc"])   

        return best_acc, eval_dict

    def _train(
        self, train_loader, optimizer, criterion, epoch, total_epochs, n_passes=1
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            for pass_ in range(n_passes):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                do_cutmix = self.cutmix and np.random.rand(1) < 0.5
                if do_cutmix:
                    x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                    logit = self.model(x)
                    loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                        logit, labels_b
                    )
                else:
                    logit = self.model(x)
                    loss = criterion(logit, y)
                _, preds = logit.topk(self.topk, 1, True, True)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)

        n_batches = len(train_loader)

        return total_loss / n_batches, correct / num_data

    def evaluation_ext(self, test_list):
        # # evaluation from out of class
        test_dataset = ImageDataset(
            pd.DataFrame(test_list),
            dataset=self.dataset,
            transform=self.test_transform,
        )
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=32, num_workers=2
        )
        eval_dict = self.evaluation(test_loader, self.criterion)

        return eval_dict

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
    
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x)

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret


    def _interpret_pred(self, y, pred):
        # # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects


    ### ---> Different Sampling Strategies <--- ###

    def rnd_sampling(self, samples):
        random.shuffle(samples)
        return samples[: self.memory_size]

    def reservoir_sampling(self, samples):
        for sample in samples:
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.memory_list[j] = sample
            # print("size of memory", len(self.memory_list))
            self.seen += 1
            

    # # # --- Prototype sampling ---
    def mean_feature_sampling(self, exemplars, samples, num_class):
        """Prototype sampling

        Args:
            features ([Tensor]): [features corresponding to the samples]
            samples ([Datalist]): [datalist for a class]

        Returns:
            [type]: [Sampled datalist]
        """

        def _reduce_exemplar_sets(exemplars, mem_per_cls):
            if len(exemplars) == 0:
                return exemplars

            exemplar_df = pd.DataFrame(exemplars)
            ret = []
            for y in range(self.num_learned_class):
                cls_df = exemplar_df[exemplar_df["label"] == y]
                ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                    orient="records"
                )

            num_dups = pd.DataFrame(ret).duplicated().sum()
            if num_dups > 0:
                logger.warning(f"Duplicated samples in memory: {num_dups}")

            return ret

        mem_per_cls = self.memory_size // num_class
        exemplars = _reduce_exemplar_sets(exemplars, mem_per_cls)
        old_exemplar_df = pd.DataFrame(exemplars)

        new_exemplar_set = []
        sample_df = pd.DataFrame(samples)
        for y in range(self.num_learning_class):
            cls_samples = []
            cls_exemplars = []
            if len(sample_df) != 0:
                cls_samples = sample_df[sample_df["label"] == y].to_dict(
                    orient="records"
                )
            if len(old_exemplar_df) != 0:
                cls_exemplars = old_exemplar_df[old_exemplar_df["label"] == y].to_dict(
                    orient="records"
                )

            if len(cls_exemplars) >= mem_per_cls:
                new_exemplar_set += cls_exemplars
                continue

            # # Assign old exemplars to the samples
            cls_samples += cls_exemplars
            if len(cls_samples) <= mem_per_cls:
                new_exemplar_set += cls_samples
                continue

            features = []
            self.feature_extractor.eval()
            with torch.no_grad():
                for data in cls_samples:
                    image = PIL.Image.open(
                        os.path.join("dataset", self.dataset, data["filename"])
                    ).convert("RGB")
                    x = self.test_transform(image).to(self.device)
                    feature = (
                        self.feature_extractor(x.unsqueeze(0)).detach().cpu().numpy()
                    )
                    feature = feature / np.linalg.norm(feature, axis=1)  # Normalize
                    features.append(feature.squeeze())

            features = np.array(features)
            logger.debug(f"[Prototype] features: {features.shape}")

            # # do not replace the existing class mean
            if self.class_mean[y] is None:
                cls_mean = np.mean(features, axis=0)
                cls_mean /= np.linalg.norm(cls_mean)
                self.class_mean[y] = cls_mean
            else:
                cls_mean = self.class_mean[y]
            assert cls_mean.ndim == 1

            phi = features
            mu = cls_mean
            # # select exemplars from the scratch
            exemplar_features = []
            num_exemplars = min(mem_per_cls, len(cls_samples))
            for j in range(num_exemplars):
                S = np.sum(exemplar_features, axis=0)
                mu_p = 1.0 / (j + 1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p, axis=1, keepdims=True)

                dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                i = np.argmin(dist)

                new_exemplar_set.append(cls_samples[i])
                exemplar_features.append(phi[i])

                # Avoid to sample the duplicated one.
                del cls_samples[i]
                phi = np.delete(phi, i, 0)

        return new_exemplar_set


    # # # --- Uncertainty sampling ---
    def uncertainty_sampling(self, samples, num_class):
        """uncertainty based sampling

        Args:
            samples ([list]): [training_list + memory_list]
        """
        self.montecarlo(samples, uncert_metric=self.uncert_metric)

        sample_df = pd.DataFrame(samples)
        mem_per_cls = self.memory_size // num_class

        ret = []
        for i in range(num_class):
            cls_df = sample_df[sample_df["label"] == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.filename.isin(pd.DataFrame(ret).filename)]
                .sample(n=num_rest_slots)
                .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).filename.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        batch_size = 32
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = ImageDataset(
            infer_df, dataset=self.dataset, transform=infer_transform
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        )

        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["image"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu()
                
                for i, cert_value in enumerate(logit):
                    sample = infer_list[batch_size * n_batch + i]
                    sample[uncert_name] = 1 - cert_value
                

    def montecarlo(self, candidates, uncert_metric="vr"):
        transform_cands = []
        # logger.info(f"Compute uncertainty by {uncert_metric}!")
        if uncert_metric == "vr":
            transform_cands = [
                Cutout(size=8),
                Cutout(size=16),
                Cutout(size=24),
                Cutout(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomRotation(90),
                Invert(),
                Solarize(v=128),
                Solarize(v=64),
                Solarize(v=32),
            ]
        elif uncert_metric == "vr_randaug":
            for _ in range(12):
                transform_cands.append(RandAugment())
        elif uncert_metric == "vr_cutout":
            transform_cands = [Cutout(size=16)] * 12
        elif uncert_metric == "vr_autoaug":
            transform_cands = [select_autoaugment(self.dataset)] * 12

        n_transforms = len(transform_cands)

        for idx, tr in enumerate(transform_cands):
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")

        for sample in candidates:
            self.variance_ratio(sample, n_transforms)

    def variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()
        

    def equal_class_sampling(self, samples, num_class):
        mem_per_cls = self.memory_size // num_class
        sample_df = pd.DataFrame(samples)
        # Warning: assuming the classes were ordered following task number.
        ret = []
        for y in range(self.num_learning_class):
            cls_df = sample_df[sample_df["label"] == y]
            ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                orient="records"
            )

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.filename.isin(pd.DataFrame(ret).filename)]
                .sample(n=num_rest_slots)
                .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).filename.duplicated().sum()
        
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret
    

    # # # --- Reservoir sampling ---
    def reservoir_sampling_with_logits(self, samples):
        
        for sample in samples:            
            
            if len(self.replay_list) < self.memory_size:        
                self.replay_list.append(sample)
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:                    
                    self.replay_list[j] = sample                   
            
            self.seen += 1
            # print("Memory updated.")

    def rnd_sampling_with_logits(self, samples):
        random.shuffle(samples)
        return samples[: self.memory_size]


    def _initialize_mean(self, samples):
        sample_df = pd.DataFrame(samples)
        incoming_classes = sample_df["label"].unique().tolist() 

        for y in incoming_classes:
            cls_samples = sample_df[sample_df["label"] == y].to_dict(
                    orient="records")
            features = []
            for sample in cls_samples:
                sample_label = sample["label"]
                sample_class = sample["klass"]
                sample_name = sample["filename"]

                self.model.eval()
                with torch.no_grad():
                    image = PIL.Image.open(
                        os.path.join("dataset", self.dataset, sample["filename"])
                    ).convert("RGB")
                    x = self.test_transform(image).to(self.device)
                    feature = (
                        self.model.features(x.unsqueeze(0)).detach().cpu().numpy()
                    )
                    feature = feature / np.linalg.norm(feature, axis=1)  # Normalize
                    # print(feature.shape)
                    features.append(feature.squeeze())

                    sample = {
                        "klass": sample_class,
                        "filename": sample_name,
                        "label": sample_label,
                        "feature": feature
                    }

                self.replay_list += [sample]

            self.cls_counter[y] += len(cls_samples)
            self.cls_mean[y] = np.mean(features, axis=0)


    # # # # --- Online Mean-feature sampling (OMFS) ---
    # def memory_update(self, cur_iter, exemplars, samples):

    #     """
    #     Initialize class_mean (& cls_counter) for the very first batch only
    #     """
    #     if len(self.replay_list) == 0:
    #         self._initialize_mean(samples)
    #         return self.replay_list
        
    #     """
    #     Check if Memory contains more than allowed no. of class samples; if yes remove one
    #     """
    #     # # Allowed per class samples for each Task iteration
    #     # samples_per_cls = self.memory_size // len((cur_iter+1)*2)
    #     samples_per_cls = self.memory_size // len(self.seen_classes)


    #     for sample in samples:
    #         # # Get 'sample' information
    #         sample_label = sample["label"] 
    #         sample_class = sample["klass"]
    #         sample_name = sample["filename"]

    #         old_exemplar_df = pd.DataFrame(self.replay_list)
    #         label_exemplars = old_exemplar_df[old_exemplar_df["label"] == sample_label].to_dict(
    #                 orient="records"
    #             )
            
    #         """
    #         get 'feature' of the 'sample'
    #         """
    #         self.model.eval()
    #         with torch.no_grad():
    #             image = PIL.Image.open(
    #                 os.path.join("dataset", self.dataset, sample["filename"])
    #             ).convert("RGB")
    #             x = self.test_transform(image).to(self.device)
    #             feature = (
    #                 self.model.features(x.unsqueeze(0)).detach().cpu().numpy()
    #             )
    #             feature = feature / np.linalg.norm(feature, axis=1)  # Normalize
           
    #         sample = {
    #                     "klass": sample_class,
    #                     "filename": sample_name,
    #                     "label": sample_label,
    #                     "feature": feature
    #                 }
            
    #         v = self.cls_mean[sample_label] # get 'class_mean' of class indexed by label 'sample_label'

    #         n = self.cls_counter[sample_label] # get 'class_counter' of class indexed by label 'sample_label'

    #         """
    #         update the class_mean online
    #         """
            
    #         self.cls_mean[sample_label] = (n/(n+1)*v) + (1/(n+1)*feature.squeeze())

    #         if cur_iter == 0:
    #             if (len(label_exemplars)) < samples_per_cls: # and len(self.replay_list) < self.memory_size:
    #                 """
    #                 When there's space for sample belong to class 'sample_label'
    #                 """
                                
    #                 self.replay_list += [sample]

    #             else:   
                
    #                 """
    #                 Replace 'sample' with the farthest sample from class_mean in 'replay_list' 
    #                 """
    #                 dist = np.linalg.norm(self.cls_mean[sample_label]-feature.squeeze())
    #                 max_dist = dist
    #                 max_idx = -1
    #                 found = False

    #                 for i, ex in enumerate(self.replay_list):
    #                     if ex["label"] == sample_label:
    #                         ex_feature = ex["feature"]
    #                         ex_label = ex["label"]
    #                         ex_fn = ex["filename"]
    #                         dst = np.linalg.norm(self.cls_mean[ex_label]-ex_feature.squeeze())

    #                         if dst > max_dist:                                
    #                             found = True
    #                             max_dist = dst
    #                             max_idx = i
                        
    #                 if max_idx != -1 and found == True:
                        
    #                     self.replay_list[max_idx] = sample
                        
    #         else:
                
    #             if len(label_exemplars) == samples_per_cls:
    #                 dist = np.linalg.norm(self.cls_mean[sample_label]-feature.squeeze())
    #                 max_dist = dist
    #                 max_idx = -1
    #                 found = False

    #                 for i, ex in enumerate(self.replay_list):
    #                     if ex["label"] == sample_label:
    #                         ex_feature = ex["feature"]
    #                         ex_label = ex["label"]
    #                         ex_fn = ex["filename"]
    #                         dst = np.linalg.norm(self.cls_mean[ex_label]-ex_feature.squeeze())

    #                         if dst > max_dist:                                
    #                             found = True
    #                             max_dist = dst
    #                             max_idx = i
                        
    #                 if max_idx != -1 and found == True:
                        
    #                     self.replay_list[max_idx] = sample

    #             else:

    #                 df = pd.DataFrame(self.replay_list)
    #                 cls_count = df.klass.value_counts().to_list()
    #                 feq_cls = max(cls_count)
    #                 feq_classes = []
    #                 exemplar_df = pd.DataFrame(self.replay_list)
                    
    #                 for y_value in self.seen_classes: 
    #                     y_exemplars = exemplar_df[exemplar_df["label"] == y_value].to_dict(
    #                             orient="records"
    #                         )
    #                     if feq_cls == len(y_exemplars):
    #                         feq_classes.append(y_value)
                    
    #                 candidates = []
    #                 max_idx = -1
    #                 for y_value in feq_classes:                    
                        
    #                     sample_dist = np.linalg.norm(self.cls_mean[sample_label]-feature.squeeze())                                               
    #                     candidate = {}
                        
    #                     found = False
    #                     max_dist = 0.0
    #                     for i, ex in enumerate(self.replay_list):
                            
    #                         if ex["label"] == y_value:
    #                             ex_feature = ex["feature"]
    #                             ex_label = ex["label"]
    #                             ex_fn = ex["filename"]
    #                             dst = np.linalg.norm(self.cls_mean[ex_label]-ex_feature.squeeze())

    #                             if dst > max_dist:
    #                                 found = True
    #                                 max_idx = i
    #                                 max_dist = dst
    #                                 candidate = {
    #                                     "filename": ex_fn,
    #                                     "label": ex_label,
    #                                     "feature": ex_feature
    #                                 }
    #                     if found == True:
    #                         candidates.append((max_idx, candidate, max_dist))
                
    #                 if max_idx != -1:
                    
    #                     self.replay_list[candidates[0][0]] = sample
                    

    #         """
    #         increment class_counter by 1
    #         """

    #         self.cls_counter[sample_label] += 1


    def memory_update(self, cur_iter, exemplars, samples):

        # --------------------------------------------------
        # First batch: initialize means and memory
        # --------------------------------------------------
        if len(self.replay_list) == 0:
            self._initialize_mean(samples)
            return self.replay_list

        # --------------------------------------------------
        # Process incoming samples one by one
        # --------------------------------------------------
        for sample in samples:

            sample_label = sample["label"]
            sample_class = sample["klass"]
            sample_name = sample["filename"]

            # ---- Extract feature
            self.model.eval()
            with torch.no_grad():
                image = PIL.Image.open(
                    os.path.join("dataset", self.dataset, sample_name)
                ).convert("RGB")
                x = self.test_transform(image).to(self.device)
                feature = self.model.features(x.unsqueeze(0))
                feature = feature.detach().cpu().numpy()
                feature = feature / np.linalg.norm(feature, axis=1)

            sample = {
                "klass": sample_class,
                "filename": sample_name,
                "label": sample_label,
                "feature": feature
            }

            # --------------------------------------------------
            # Update class mean (online)
            # --------------------------------------------------
            n = self.cls_counter[sample_label]
            mu = self.cls_mean[sample_label]
            self.cls_mean[sample_label] = (n / (n + 1)) * mu + (1 / (n + 1)) * feature.squeeze()
            self.cls_counter[sample_label] += 1

            # --------------------------------------------------
            # CASE 1: Memory NOT full → always add
            # --------------------------------------------------
            if len(self.replay_list) < self.memory_size:
                self.replay_list.append(sample)
                continue

            # --------------------------------------------------
            # CASE 2: Memory FULL → apply OMFS
            # --------------------------------------------------

            # ---- Compute per-class quota (current stage)
            samples_per_cls = self.memory_size // len(self.seen_classes)

            replay_df = pd.DataFrame(self.replay_list)
            label_exemplars = replay_df[replay_df["label"] == sample_label]

            # --------------------------------------------------
            # 2(a) Class under-allocated → replace from largest class
            # --------------------------------------------------
            if len(label_exemplars) < samples_per_cls:

                class_counts = replay_df.label.value_counts()
                majority_class = class_counts.idxmax()

                candidates = replay_df[replay_df["label"] == majority_class]

                # Remove farthest-from-mean sample in majority class
                max_dist = -1
                remove_idx = None

                for idx, ex in candidates.iterrows():
                    ex_feat = ex["feature"].squeeze()
                    dist = np.linalg.norm(self.cls_mean[majority_class] - ex_feat)
                    if dist > max_dist:
                        max_dist = dist
                        remove_idx = idx

                if remove_idx is not None:
                    self.replay_list.pop(remove_idx)
                    self.replay_list.append(sample)

            # --------------------------------------------------
            # 2(b) Class meets/exceeds quota → herding
            # --------------------------------------------------
            else:
                # Distance of incoming sample
                incoming_dist = np.linalg.norm(
                    self.cls_mean[sample_label] - feature.squeeze()
                )

                max_dist = incoming_dist
                remove_idx = None

                for i, ex in enumerate(self.replay_list):
                    if ex["label"] == sample_label:
                        ex_feat = ex["feature"].squeeze()
                        dist = np.linalg.norm(self.cls_mean[sample_label] - ex_feat)
                        if dist > max_dist:
                            max_dist = dist
                            remove_idx = i

                if remove_idx is not None:
                    self.replay_list[remove_idx] = sample

        return self.replay_list


    
    # # # --- Random sampling ---
    def random_sampling(self, memory, stream_batch):
        if len(memory) + len(stream_batch) <= self.memory_size:
            for sample in stream_batch:
                memory.append(sample)
        else:
            temp = memory + stream_batch
            random.shuffle(temp)
            memory = temp[:self.memory_size]
        
        self.replay_list = memory

    
    # # # --- Class balanced reservoir sampling ---
    def reservoir_sampling_balanced(self, stream, memory, num_classes):
   
        # stream_samples = []
    
        # for inputs, labels in stream:
        #     inputs = torch.split(inputs, 1)
        #     labels = torch.split(labels, 1)
        #     for i, l in zip(inputs, labels):        
        #         stream_samples.append((i,l))
                    
        for sample in stream:
            if len(memory) < self.memory_size:
                memory.append(sample)
                self.class_counts[sample["label"]] += 1
            else:
                if self.class_counts[sample["label"]] < self.memory_size / num_classes:
                    # # Add new sample and remove a sample from the largest class
                    largest_class = max(self.class_counts, key=self.class_counts.get)
                    largest_class_indices = [i for i, itm in enumerate(memory) if itm["label"] == largest_class]
                    replace_idx = random.choice(largest_class_indices)
                    memory[replace_idx] = (sample)
                    self.class_counts[largest_class] -= 1
                    self.class_counts[sample["label"]] += 1
                else:
                    # # Reservoir sampling step
                    mc = self.class_counts[sample["label"]]
                    # total_seen = sum(self.class_counts)               
                    # total_seen = sum(self.class_counts.values())
                    # total_seen = self.class_counts[y.item()]

                    nc = sum(1 for label in stream if label["label"] == sample["label"])  # Number of stream instances of class c ≡ y encountered thus far
                    u = random.uniform(0, 1)
                    
                    if u <= (mc/nc): # random.random() < (mc / total_seen)
                        # # Replace a random instance of the same class
                        class_indices = [i for i, itm in enumerate(memory) if itm["label"] == sample["label"]]
                        if class_indices:
                            replace_idx =  random.choice(class_indices)
                        # else:
                        #     replace_idx = random.randint(0, len(memory) - 1)
                        
                            memory[replace_idx] = sample

        return memory 


    def update_class_prototypes(self, dataset):
        """
        Incrementally update class prototypes from a dataset.
        Each sample is a dict with batched 'image' and 'label'.
        """
        with torch.no_grad():
            for sample in dataset:
                x = sample["image"].to(self.device)    # [B, C, H, W]
                y = sample["label"]                    # [B]

                feats = self.model.features(x)         # [B, D, 1, 1] or similar
                feats = torch.flatten(feats, start_dim=1).cpu()  # [B, D]

                for i in range(len(y)):
                    cls = int(y[i])
                    feat = feats[i]

                    if cls not in self.class_prototypes:
                        self.class_prototypes[cls] = {"sum": feat.clone(), "count": 1, "mean": feat.clone()}
                    else:
                        self.class_prototypes[cls]["sum"] += feat
                        self.class_prototypes[cls]["count"] += 1
                        cnt = self.class_prototypes[cls]["count"]
                        self.class_prototypes[cls]["mean"] = self.class_prototypes[cls]["sum"] / cnt
                    
                    self.seen_labels.add(cls)



    # def select_hard_negatives(self, model, anchor_dataset, memory_dataset, batch_size=10, device="cpu"):
    #     """
    #     Selects hard negatives from memory_dataset relative to anchor_dataset using prototypes.
    #     Each sample is expected to be a dict with 'image' and 'label'.

    #     Returns:
    #         List[dict]: Balanced list of hard negative samples from memory_dataset
    #     """
    #     # Step 1: Collect anchor class IDs
    #     anchor_labels = {int(sample["label"]) for sample in anchor_dataset}
    #     memory_labels = set(self.class_prototypes.keys())

    #     # Step 2: Extract prototypes for anchor classes
    #     anchor_protos = {
    #         c: self.class_prototypes[c]["mean"].to(device)
    #         for c in anchor_labels if c in self.class_prototypes
    #     }

    #     # Step 3: Find candidate classes not in anchor set
    #     candidate_classes = list(memory_labels - anchor_labels)
    #     class_dists = []

    #     for mem_class in candidate_classes:
    #         mem_proto = self.class_prototypes[mem_class]["mean"].to(device)

    #         # Compute cosine similarity with each anchor prototype
    #         min_dist = min(
    #             F.cosine_similarity(mem_proto.unsqueeze(0), anchor_protos[ac].unsqueeze(0), dim=1).item()
    #             for ac in anchor_protos
    #         )
    #         class_dists.append((mem_class, min_dist))

    #     # Step 4: Sort by similarity (descending = harder negatives)
    #     class_dists.sort(key=lambda x: -x[1])  # more similar = harder

    #     # Step 5: Choose up to batch_size classes
    #     num_classes = min(len(class_dists), batch_size)
    #     selected_classes = [cid for cid, _ in class_dists[:num_classes]]

    #     # Step 6: Collect samples from selected classes
    #     class_to_samples = defaultdict(list)
    #     for sample in memory_dataset:
    #         class_to_samples[int(sample["label"])].append(sample)

    #     n_per_class = max(1, batch_size // num_classes)
    #     selected = []

    #     for cls in selected_classes:
    #         samples = class_to_samples[cls]
    #         selected.extend(samples[:n_per_class])  # deterministic: take first N
    #         if len(selected) >= batch_size:
    #             break

    #     # Step 7: Trim to exact batch size if overfilled
    #     selected = selected[:batch_size]

    #     return selected
    

    # def compute_class_prototypes(self, samples, model, device='cuda:1'):
    #     """
    #     Compute class prototypes using encoded features.

    #     Args:
    #         samples (list of dict): each with 'image' [3, 32, 32], and 'label'
    #         model: encoder model that outputs normalized features
    #         device: 'cuda' or 'cpu'

    #     Returns:
    #         dict: {class_label: prototype tensor}
    #         list: [(sample_dict, distance to prototype), ...]
    #     """
    #     model.eval()
    #     model.to(device)

    #     class_features = defaultdict(list)
    #     sample_features = []

    #     with torch.no_grad():
    #         for sample in samples:
    #             img = sample['image'].unsqueeze(0).to(device)
    #             feat = model.features(img)  # [1, dim]
                
    #             feat = F.normalize(feat.squeeze(0), p=2, dim=0)
    #             label = sample['label']

    #             class_features[label].append(feat)
    #             sample_features.append((sample, feat))

    #     # Compute prototype per class
    #     self.prototypes = {
    #         cls: torch.stack(feats).mean(dim=0)
    #         for cls, feats in class_features.items()
    #     }

    #     return sample_features
