### Continual Learner with Anytime Inference (CLAI) ###

import logging
import random

import numpy as np
import pandas as pd

# from collections import defaultdict

import torch
import torch.nn as nn 
# from torchvision import transforms
# import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.finetune import Finetune

from utils.data_loader import ImageDataset
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, Resize
# from methods.angle_similar_loss import RKAngle
# from utils.utils import AverageMeter

from tsne import extract_features, plot_tsne
from utils.utils import evaluate_accuracy, AAUCTracker
from utils.utils import extract_memory_features, plot_tsne_subset_classes, l2_normalize, induce_dirichlet_imbalance


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


# # # --- Settings the warnings to be ignored ---
import warnings 
warnings.filterwarnings('ignore')


class CLAI(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        # # # ---> Default experiemntal setting for CLAI <---
        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.exp_env = kwargs["stream_env"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "reservoir"
        if kwargs["dataset"] == "cifar10" or kwargs["dataset"] == "cifar100":
            inp_size = 32
        elif kwargs["dataset"] == "miniimagenet":
            inp_size = 84
        elif kwargs["dataset"] == "tinyimagenet":
            inp_size = 64
        elif kwargs["dataset"] == "core50":
            inp_size = 128
        else:
            raise ValueError("Dataset not supported. Choose from [cifar10, cifar100, miniimagenet, tinyimagenet, core50]")

        # # -- Data transformations ---
        self.transform = nn.Sequential(
        # Resize(84, 84),
        RandomResizedCrop(size=(inp_size, inp_size), scale=(0.2, 1.)), # 32 for CIFAR10 and CIFAR100, 84 for MINI-IMAGENET
        RandomHorizontalFlip(),
        ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        RandomGrayscale(p=0.2)
        )

        # delta_n = 10  # mini-batch size
        self.aauc_tracker = AAUCTracker(delta_n=1000)
        # self.dist_loss = RKAngle()
    
    # # # ---> TRAINING Function <---
    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=0):
        # # Get TRAIN and TEST datalist for the current task
        train_list = self.streamed_list 
        test_list = self.test_list        
        train_list = induce_dirichlet_imbalance(
            samples=train_list,
            alpha=0.5,
            seed=0
        )
        random.shuffle(train_list)

        # # Get TEST loader for the current task (all seen classes so far)
        _, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        # # Get train minibatches for the current task TRAIN data
        train_minibatch = self.get_train_minibatch(batch_size=batch_size, trainlist=train_list)
        
        
        # # Get classes  for the current task
        _dataset = ImageDataset(
            pd.DataFrame(train_list),
            dataset=self.dataset,
            transform=self.test_transform,
        )
        new_task_classes = set(_dataset.data_frame["label"].unique())
        self.task_classes[cur_iter] = list(new_task_classes)


        # # Print out the statistics
        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.replay_list)}")
        logger.info(f"Train samples: {len(train_list)+len(self.replay_list)}")
        logger.info(f"Test samples: {len(test_list)}")



        # # # ---> TRAIN <---
        best_acc = 0.0
        eval_dict = dict()
        self.model = self.model.to(self.device)    

        # # Training loop
        for epoch in range(n_epoch):
            # # initialize for each task
            # if epoch <= 0:  # Warm start of 1 epoch
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = self.lr * 0.1
            # elif epoch == 1:  # Then set to maxlr
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = self.lr
            # else:  # Aand go!
            #     self.scheduler.step()

            train_loss = self._train(cur_iter, 
                                        batch_size=batch_size, 
                                        train_minibatch_data=train_minibatch,
                                        optimizer=self.optimizer,
                                        criterion=self.criterion
                                    )
            
            # logger.info(f"Classifier training after task {cur_iter+1}")
            # full_data = self.replay_list #+ train_list
            # random.shuffle(full_data)
            # full_dataset = ImageDataset(
            #         pd.DataFrame(full_data),
            #         dataset=self.dataset,
            #         transform=self.test_transform,
            #     )
            
            # full_loader = DataLoader(full_dataset,
            #         shuffle=True, 
            #         batch_size=50,
            #         num_workers=0,
            #         drop_last=True, 
            #     )
            
            # for ep in range(100):
            #     self._classifier_train(cur_iter,
            #             batch_size=batch_size,
            #             train_data=full_loader,
            #             optimizer=self.optimizer_fc,
            #             criterion=self.classifier_criterion
            #         )
            
            # writer.add_scalar(f"task{cur_iter}/FE_train/loss", train_loss, epoch)
            # # # writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)            
            # writer.add_scalar(
            #     f"task{cur_iter}/FE_train/lr", self.optimizer.param_groups[0]["lr"], epoch
            # )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | FE_train_loss {train_loss:.4f} | " #train_acc {train_acc:.4f} | "
                # f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            # eval_dict = self.evaluate(test_loader=test_loader)

            if cur_iter == 0:
                eval_dict = self.validate(test_loader=test_loader,
                                                criterion=self.classifier_criterion
                                            )
            else:
                eval_dict = self.validate_(test_loader=test_loader,
                                            criterion=self.classifier_criterion,
                                            new_task_classes=list(new_task_classes))
                
            # final_aauc = self.aauc_tracker.compute()
            # print(f"AAUC = {final_aauc:.4f}")

            
        # writer.add_scalar(f"task{cur_iter}/classifier_train/loss", classifier_train_loss, epoch)
        # writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
        # writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)

        logger.info(
            f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | " # Classifier_train_loss {classifier_train_loss:.4f} | " #train_acc {train_acc:.4f} | "
            f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
            f"lr {self.optimizer_fc.param_groups[0]['lr']:.4f}"
        )

        best_acc = max(best_acc, eval_dict["avg_acc"])


        # # # ---> Show Memory content <---
        df = pd.DataFrame(self.replay_list)  
        logger.info("Memory Statistics after task ")
        logger.info(f"\n{df.klass.value_counts(sort=True)}")  

        if cur_iter > 0:
            logger.info(f"Prediction bias after training on task {cur_iter + 1}: {eval_dict['old_to_new_misclass_pct']:.4f}")


        torch.save(self.model, "./clai_model.pt") 

        if cur_iter == 4:
            # class_to_task = build_class_to_task(self.task_classes)
            model = torch.load(f"./clai_model.pt")
            m_dataset = ImageDataset(
                pd.DataFrame(self.replay_list),
                dataset=self.dataset,
                transform=self.test_transform,
            )
            
            m_loader = DataLoader(m_dataset,
                shuffle=False, #important
                batch_size=100,
                num_workers=0, 
            )
            features, class_labels = extract_memory_features(
                model,
                m_loader,
                self.device
            )

            features = l2_normalize(features)

            # Good qualitative plot
            # selected_classes = np.random.choice(np.arange(50), 10, replace=False).tolist()
            np.random.seed(0)
            selected_classes = np.random.choice(
                np.arange(50),
                10,
                replace=False
            ).tolist()

            plot_tsne_subset_classes(
                features,
                class_labels,
                selected_classes,
                # title="t-SNE of Replay Memory (Class-wise)",
                save_path="./figures/tsne_supcon_0.5.png"
            ) 

            exit()  
        # if cur_iter == 4:
        #     model = torch.load("./clai_model.pt")
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

        #     # # Map each class to its task ID
        #     # class_to_task = {}
        #     # for task_id, class_list in self.task_classes.items():
        #     #     for cls in class_list:
        #     #         class_to_task[cls] = task_id


        #     features, logits, labels = extract_features(model, m_loader, self.device, classifier=self.classifier)
        #     # features, logits, class_labels, task_labels = extract_features(
        #     #     model, data_loader=m_loader, device=self.device, classifier=self.classifier, class_to_task=class_to_task
        #     # )

        #     # plot_tsne(features, task_labels, title="t-SNE: Feature Embeddings by Task")

        #     # # Apply t-SNE to reduce the dimensionality of the features
        #     # tsne_l = TSNE(n_components=2, 
        #     #             n_iter=1000, 
        #     #             perplexity=50,  # higher value for better global structure
        #     #             learning_rate=500,  # adjusted learning rate
        #     #             random_state=42)
        #     # logits_tsne = tsne_l.fit_transform(logits)

        #     # # Plot the t-SNE results
        #     # plt.figure(figsize=(10, 8))
        #     # sns.scatterplot(x=logits_tsne[:, 0], y=logits_tsne[:, 1], hue=labels, palette='tab10', s=60, alpha=0.7, legend=False)
        #     # # plt.title("t-SNE plot of ResNet-18 Logits CLAI")
        #     # # plt.xlabel("t-SNE Component 1")
        #     # # plt.ylabel("t-SNE Component 2")
        #     # # plt.legend(title="Classes")
        #     # plt.savefig("tsne_logits_clai_cifar100.png")
        #     # plt.show() 

        #     tsne_f = TSNE(n_components=2, 
        #                 n_iter=1000, 
        #                 perplexity=50,  # higher value for better global structure
        #                 learning_rate=500,  # adjusted learning rate
        #                 random_state=42)
        #     features_tsne = tsne_f.fit_transform(features)

        #     # Plot the t-SNE results
        #     plt.figure(figsize=(10, 8))
        #     sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=labels, palette='tab10', s=60, alpha=0.7)
        #     plt.title("(b) CLAI", fontsize=20)
        #     # plt.xlabel("t-SNE Component 1")
        #     # plt.ylabel("t-SNE Component 2")
        #     # plt.legend(title="Classes")
        #     plt.savefig("tsne_CLAI_cifar10.png")
        #     plt.show()    


        return best_acc, eval_dict
    

    # # # ---> TRAINING Helper Function <---
    def _train(
        self, cur_iter, batch_size, train_minibatch_data, optimizer, criterion
    ):
        total_loss = 0.0
        self.model.train()

        for i in range(len(train_minibatch_data)):

            incoming_cls_df = pd.DataFrame(train_minibatch_data[i])
            incoming_classes = incoming_cls_df["label"].unique().tolist()
            # self.seen_classes = list(set(self.seen_classes + incoming_classes))
            self.seen_classes = sorted(
                set(self.seen_classes).union(incoming_classes)
            )


            self.model.train()

            minibatch_dataset = ImageDataset(
                    pd.DataFrame(train_minibatch_data[i]),
                    dataset=self.dataset,
                    transform=self.test_transform,
                )
            
            minibatch_loader = DataLoader(minibatch_dataset,
                    shuffle=True, 
                    batch_size=batch_size,
                    num_workers=0, 
                )

            for data in minibatch_loader:
                
                x_train = data["image"]
                y_train = data["label"]                

                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                if len(self.replay_list) >= 10:
                    
                    minibatch_memory = ImageDataset(
                            pd.DataFrame(random.sample(self.replay_list, 10)),
                            dataset=self.dataset,
                            transform=self.test_transform
                        )

                    minibatch_memloader = DataLoader(minibatch_memory,
                            shuffle=True, 
                            batch_size=10,
                            num_workers=0,
                        )
                    
                    for mem_data in (minibatch_memloader):
                        mem_x = mem_data["image"]
                        mem_y = mem_data["label"]

                        mem_x = mem_x.to(self.device)
                        mem_y = mem_y.to(self.device)
                        
                        # # ***Instance-based Contrastive Learning***
                        combined_batch = torch.cat((mem_x, x_train))
                        combined_labels = torch.cat((mem_y, y_train))
                        # print("combined shape ", combined_batch.shape, combined_labels.shape)

                        combined_batch_aug = self.transform(combined_batch)
                        # print("combined aug shape ", combined_batch_aug.shape)

                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        # print("features shape ", features.shape)
                        # print(self.model.forward(combined_batch).shape)
                        
                        loss = criterion(features, combined_labels)
                     
                        # loss = 0.2 * F.mse_loss(self.model.features(self.transform(mem_x)), mem_data["feature"].to(self.device))

                        # # SGD
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # # calculate total_loss
                        total_loss += loss.item()

        
            # print(f"Training completed for minibatch {i+1}/{len(train_minibatch_data)}")
            # # ---> TRAIN the Classifier ONLY with Memory data <---
            if len(self.replay_list) >= 50: 
                # logger.info("Classifier training with Memory data...")               
                memory_dataset = ImageDataset(
                        pd.DataFrame(random.sample(self.replay_list, 50)),
                        dataset=self.dataset,
                        transform=self.test_transform,
                    )
                
                memory_loader = DataLoader(memory_dataset,
                        shuffle=True, 
                        batch_size=50,
                        num_workers=0,
                        drop_last=True, 
                    )
                
                self._classifier_train(cur_iter,
                        batch_size=batch_size,
                        train_data=memory_loader,
                        optimizer=self.optimizer_fc,
                        criterion=self.classifier_criterion
                    )
            
            # # # --- Sampling for populating memory ---
            # logger.info("Memory Update...")

            # self.replay_list = self.rnd_sampling_with_logits(train_minibatch_data[i]+self.replay_list)     
            self.reservoir_sampling_with_logits(train_minibatch_data[i])
            # self.memory_update(cur_iter, self.replay_list, train_minibatch_data[i])
            # self.random_sampling(self.replay_list, train_minibatch_data[i])
            # self.replay_list = self.reservoir_sampling_balanced(train_minibatch_data[i], self.replay_list, len(self.seen_classes))
            
            # logger.info("Memory Update Completed.")
            # logger.info(f"Memory size: {len(self.replay_list)}")
            # print(i)
            # if i%100 == 0 and i > 0:
            #     print("Acc computed")
            #     acc = evaluate_accuracy(
            #         model=self.model,
            #         classifier=self.classifier,
            #         test_data=self.test_list,   # full test list
            #         seen_classes=self.seen_classes,
            #         device=self.device
            #     )
            #     self.aauc_tracker.update(acc)
            
            n_batches = len(train_minibatch_data)

        return total_loss / n_batches #, correct / num_data
    

    # # # ---> TRAINING Helper Function for Classifier <---
    def _classifier_train(self, cur_iter, batch_size, train_data, optimizer, criterion):
        
        self.model.eval()
        self.classifier.train()

        for data in train_data:
            
            x_train = data["image"]
            y_train = data["label"]

            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)

            with torch.no_grad():
                feature = self.model.features(x_train)
                # feature.data = feature.data / feature.data.norm()
            # output = self.classifier(feature.detach())
            output = self.classifier(feature.squeeze().detach())
            
            loss = criterion(output, y_train)

            # # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # # # ---> EVALUATION Functions <---
    def validate(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
    
        self.model.eval()
        self.classifier.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                # x = x.data / x

                # # forward
                feature = self.model.features(x)
             
                # feature.data = feature.data / feature.data.norm()
                output = self.classifier(feature)
                loss = criterion(output, y)
                
                pred = torch.argmax(output, dim=-1)
                _, preds = output.topk(self.topk, 1, True, True)

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
    
    def validate_(self, test_loader, criterion, new_task_classes):
        """
        Validate model and compute:
        - Average accuracy
        - Per-class accuracy
        - % of old task samples misclassified as new task classes

        Args:
            test_loader: DataLoader with test samples.
            criterion: Loss function.
            new_task_classes: List of class indices for the current (new) task.

        Returns:
            Dictionary with avg_loss, avg_acc, cls_acc, old_to_new_misclass_pct
        """
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        new_task_classes = set(new_task_classes)
        old_to_new_misclass = 0
        total_old_samples = 0

        self.model.eval()
        self.classifier.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"].to(self.device)
                y = data["label"].to(self.device)

                feature = self.model.features(x)
                output = self.classifier(feature)

                loss = criterion(output, y)
                pred = torch.argmax(output, dim=-1)
                _, preds = output.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

                # # Old → New misclassification tracking
                for true_label, predicted_label in zip(y, pred):
                    if true_label.item() not in new_task_classes:  # Old class
                        total_old_samples += 1
                        if predicted_label.item() in new_task_classes:
                            old_to_new_misclass += 1

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        if total_old_samples > 0:
            old_to_new_misclass_pct = old_to_new_misclass / total_old_samples
        else:
            old_to_new_misclass_pct = 0.0

        ret = {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "cls_acc": cls_acc,
            "old_to_new_misclass_pct": old_to_new_misclass_pct
        }

        return ret    

    # # # ---> EVALUATION Helper Function <---
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
    

    # # # ---> HELPER Function to get minibatches from streamed list <---
    def get_train_minibatch(self, batch_size, trainlist):

        """
        # Returns a list with minibatches (lists) of size 'batch_size'
        """

        train_minibatches = []
        for i in range(0, len(trainlist), batch_size):
            j = i 
            minibatch = self.streamed_list[j:j+batch_size]
            if len(minibatch) == batch_size:
                train_minibatches.append(minibatch)

        return train_minibatches
    


    # def allocate_batch_size(self, n_old_class, n_new_class):
    #     new_batch_size = int(
    #         self.batch_size * n_new_class / (n_old_class + n_new_class)
    #     )
    #     old_batch_size = self.batch_size - new_batch_size
    #     return new_batch_size, old_batch_size


    # def evaluate(self, test_loader):
    #     self.model.eval()
    #     exemplar_means = {}
    #     seen_classes = pd.DataFrame(self.replay_list)["label"].unique().tolist()
    #     cls_exemplar = {cls: [] for cls in seen_classes}

    #     replay_data = ImageDataset(pd.DataFrame(self.replay_list),
    #                                        dataset=self.dataset,
    #                                        transform=self.test_transform)
       
    #     for data in replay_data:
    #         x = data["image"]
    #         y = data["label"]
    #         cls_exemplar[y.item()].append(x)

    #     for cls, exemplar in cls_exemplar.items():
    #         features = []
    #         # Extract feature for each exemplar in p_y
    #         for ex in exemplar:
    #             ex = ex.to(self.device)
    #             feature = self.model.features(ex.unsqueeze(0)).detach().clone()
    #             feature = feature.squeeze()
    #             feature.data = feature.data / feature.data.norm()  # Normalize
    #             features.append(feature)
    #         if len(features) == 0:
    #             mu_y = torch.normal(0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size()))
    #             mu_y = mu_y.to(self.device)
    #             mu_y = mu_y.squeeze()
    #         else:
    #             features = torch.stack(features)
    #             mu_y = features.mean(0).squeeze()
    #         mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
    #         exemplar_means[cls] = mu_y
        
    #     with torch.no_grad():
    #         acc = AverageMeter()
    #         for i, batch_xy in enumerate(test_loader):
    #             batch_x = batch_xy["image"]
    #             batch_y = batch_xy["label"]

    #             batch_x = batch_x.to(self.device)
    #             batch_y = batch_y.to(self.device)


    #             if self.mode == "scr": # or self.mode == "pcr":  
    #                 feature = self.model.features(batch_x)  # (batch_size, feature_size)
                    
    #                 for j in range(feature.size(0)):  # Normalize
    #                     feature.data[j] = feature.data[j] / feature.data[j].norm()
    #                 feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
    #                 means = torch.stack([exemplar_means[cls] for cls in seen_classes])  # (n_classes, feature_size)

    #                 #old ncm
    #                 means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
    #                 means = means.transpose(1, 2)
    #                 feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
    #                 dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
    #                 _, pred_label = dists.min(1)
    #                 # may be faster
    #                 # feature = feature.squeeze(2).T
    #                 # _, preds = torch.matmul(means, feature).max(0)
    #                 correct_cnt = (np.array(seen_classes)[
    #                                     pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)

    #             elif self.mode == "pcr":
    #                 logits, _ = self.model.pcrForward(batch_x)
    #                 # mask = torch.zeros_like(logits)
    #                 # mask[:, self.old_labels] = 1
    #                 # logits = logits.masked_fill(mask == 0, -1e9)
    #                 _, pred_label = torch.max(logits, 1)
    #                 correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
    #             else:
    #                 logits = self.model.forward(batch_x)
    #                 _, pred_label = torch.max(logits, 1)
    #                 correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)

    #             acc.update(correct_cnt, batch_y.size(0))
        
    #     # print("Task ", cur_iter, " accuracy is: ", acc.avg())

    #     ret = {"avg_loss": 0.0, "avg_acc": acc.avg(), "cls_acc": [0.0]}

    #     return ret