
import random
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from utils.data_loader import ImageDataset
import pandas as pd



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

def evaluate_accuracy(model, classifier, test_data, seen_classes, device, batch_size=10):
    """
    Evaluate accuracy on test samples belonging to seen classes only (C_n).
    """

    # ---- Filter test data by C_n
    eval_data = [s for s in test_data if s["label"] in seen_classes]
    # print(
    # "[EVAL DEBUG]",
    # "seen_classes =", seen_classes,
    # "eval_size =", len(eval_data)
    # )
    # print(model)

    if len(eval_data) == 0:
        return 0.0

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i + batch_size]

            images, labels = [], []
            for s in batch:
                img = Image.open(
                    os.path.join("dataset/miniimagenet", s["filename"])
                ).convert("RGB")
                images.append(img)
                labels.append(s["label"])

            # images = torch.stack([
            #     transforms.ToTensor()(img) for img in images
            # ]).to(device)
            transform = transforms.Compose([
                transforms.Resize((84, 84)),
                # transforms.CenterCrop(84),
                transforms.ToTensor()
            ])

            images = torch.stack([
                transform(img) for img in images
            ]).to(device)


            labels = torch.tensor(labels).to(device)

            logits = model.features(images)
            logits = classifier(logits)
            # print(logits.shape)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
    # exit()
    return correct / total


class AAUCTracker:
    def __init__(self, delta_n):
        self.delta_n = delta_n
        self.acc_list = []
        self.total_samples = 0

    def update(self, acc):
        self.acc_list.append(acc)
        self.total_samples += self.delta_n
        # print("[AAUC] update, acc =", acc)


    def compute(self):
        aauc = sum(acc * self.delta_n for acc in self.acc_list)
        return aauc / self.total_samples



class DictDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

def split_balanced_test_valid_loaders(test_loader, batch_size=64, num_workers=2, seed=42):
    """
    Splits samples from test_loader into two balanced DataLoaders: test and validation.
    """
    all_samples = []

    # Step 1: Collect all samples from test_loader
    for batch in test_loader:
        images = batch["image"]
        labels = batch["label"]
        for i in range(len(images)):
            all_samples.append({
                "image": images[i],
                "label": labels[i].item()
            })

    # Step 2: Group by class
    class_to_samples = defaultdict(list)
    for sample in all_samples:
        class_to_samples[sample["label"]].append(sample)

    random.seed(seed)
    test_split, valid_split = [], []

    # Step 3: Split each class 50/50
    for samples in class_to_samples.values():
        random.shuffle(samples)
        half = len(samples) // 2
        valid_split.extend(samples[:half])
        test_split.extend(samples[half:])

    # Step 4: Wrap in DataLoaders
    test_dataset = DictDataset(test_split)
    valid_dataset = DictDataset(valid_split)

    new_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return new_test_loader, valid_loader



def get_classwise_proportions(valid_loader, cls_acc, total_samples=10, min_per_class=1):
    """
    Get fixed number of samples allocated to lowest-performing classes.

    Args:
        valid_loader: DataLoader with validation data.
        cls_acc: List[float] - class-wise accuracies.
        total_samples: int - total number of samples to select.
        min_per_class: int - minimum number to assign per class.

    Returns:
        Dict[int, int]: {class_id: num_samples_to_select}
    """
    class_counts = defaultdict(int)

    # Count examples of each class in valid_loader
    for batch in valid_loader:
        labels = batch["label"] if isinstance(batch, dict) else batch[1]
        for label in labels:
            class_counts[int(label)] += 1

    # Compute inverse accuracy
    acc_tensor = torch.tensor(cls_acc)
    inverse_acc = 1.0 - acc_tensor
    inverse_acc = torch.clamp(inverse_acc, min=1e-3)

    valid_classes = list(class_counts.keys())
    valid_inverse_acc = [(cls, inverse_acc[cls].item()) for cls in valid_classes]

    # Sort classes by lowest accuracy (highest inverse accuracy)
    sorted_classes = sorted(valid_inverse_acc, key=lambda x: -x[1])
    top_classes = [cls for cls, _ in sorted_classes[:total_samples]]

    proportions = {cls: 1 for cls in top_classes}

    remaining = total_samples - len(proportions)
    if remaining > 0:
        # Distribute remaining samples to lowest-accuracy classes
        for cls in top_classes:
            if remaining == 0:
                break
            proportions[cls] += 1
            remaining -= 1

    return proportions


def sample_from_replay(replay_list, proportions, total_samples=10):
    # Group replay_list by label
    grouped = defaultdict(list)
    for sample in replay_list:
        grouped[sample["label"]].append(sample)

    # Step 1: Cap each class count to available samples
    for label in list(proportions.keys()):
        max_available = len(grouped[label])
        if proportions[label] > max_available:
            proportions[label] = max_available

    # Step 2: Adjust to make total == total_samples
    current_total = sum(proportions.values())

    # Get all labels with remaining capacity
    while current_total < total_samples:
        # Filter for labels that can still provide more samples
        candidates = [label for label in grouped if proportions[label] < len(grouped[label])]
        if not candidates:
            raise ValueError("Not enough total samples in replay_list to select 10 unique items.")

        chosen_label = random.choice(candidates)
        proportions[chosen_label] += 1
        current_total += 1

    # Step 3: Collect the sampled dictionaries
    sampled = []
    for label, num_samples in proportions.items():
        sampled.extend(random.sample(grouped[label], num_samples))

    # print("sampled",sampled)
    
    return sampled
    

import torch
import torch.nn.functional as F
from torchvision import transforms

def select_hardest_samples_raw(sample_list, model, num_selected=10, n_views=4, device='cuda'):
    """
    Select hardest samples from raw images by generating views, encoding them,
    and ranking by intra-view distance.

    Args:
        sample_list (list of dict): each dict has 'image': Tensor [3, 32, 32], 'label': int
        model: feature encoder model (returns normalized features)
        num_selected (int): how many samples to return
        n_views (int): how many views to generate per image
        device (str): cuda or cpu

    Returns:
        list of dict: hardest samples (same format as input)
    """
    model.eval()
    model.to(device)

    # Define transform pipeline
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
    ])

    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    hardness_scores = []
    sample_list = ImageDataset(
                    pd.DataFrame(sample_list),
                    dataset="cifar100",
                    transform=test_transform,
                )
    for sample in sample_list:
        # print(sample)
        img = sample['image'].to(device)  # [3, 32, 32]
        views = torch.stack([transform(img).unsqueeze(0) for _ in range(n_views)])  # [n_views, 1, 3, 32, 32]
        views = views.squeeze(1)  # [n_views, 3, 32, 32]

        with torch.no_grad():
            feats = model.features(views)  # [n_views, feat_dim], already normalized

        # Cosine distance
        sim_matrix = torch.matmul(feats, feats.T)
        dist_matrix = 1 - sim_matrix
        dist_matrix.fill_diagonal_(float('-inf'))

        max_dist = dist_matrix.max().item()
        hardness_scores.append((max_dist, sample))

    sorted_samples = sorted(hardness_scores, key=lambda x: x[0], reverse=True)
    return [s[1] for s in sorted_samples[:num_selected]]


import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict
import random

def compute_entropy(probs):
    """Compute entropy from softmax probabilities."""
    return -(probs * probs.log()).sum(dim=1)  # shape: [n_views]

def entropy_based_diverse_sampling(
    sample_list, model, head, num_selected=10, n_views=2, per_class=True, device='cuda'
):
    """
    Select samples with diverse entropy (uncertain & moderately confident) across classes.

    Args:
        sample_list (list): each dict has 'image': Tensor [3, 32, 32], 'label': int
        model: encoder model (outputs features)
        head: classification head (outputs logits)
        num_selected (int): total samples to select
        n_views (int): number of augmentations per image
        per_class (bool): enforce class balance
        device (str): 'cuda' or 'cpu'

    Returns:
        list: selected samples (same dict format)
    """
    model.eval()
    head.eval()
    model.to(device)
    head.to(device)

    # Data augmentation for views
    aug = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    ])
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    sample_scores = []
    sample_list = ImageDataset(
                    pd.DataFrame(sample_list),
                    dataset="cifar100",
                    transform=test_transform,
                )

    for sample in sample_list:
        img = sample['image'].to(device)
        views = torch.stack([aug(img).unsqueeze(0) for _ in range(n_views)]).squeeze(1)  # [n_views, 3, 32, 32]

        with torch.no_grad():
            feats = model.features(views)  # [n_views, feat_dim]
            logits = head(feats)  # [n_views, num_classes]
            probs = F.softmax(logits, dim=1)
            entropies = compute_entropy(probs)  # [n_views]
            avg_entropy = entropies.mean().item()

        sample_scores.append({
            "sample": sample,
            "entropy": avg_entropy
        })

    # Group by class if per_class=True
    if per_class:
        class_buckets = defaultdict(list)
        for s in sample_scores:
            class_buckets[s["sample"]["label"]].append(s)

        selected = []
        per_class_quota = max(1, num_selected // len(class_buckets))

        for cls, items in class_buckets.items():
            # Sort by entropy descending
            sorted_items = sorted(items, key=lambda x: x["entropy"], reverse=True)

            # Get top N//2 and middle N//2 entropies
            top_k = sorted_items[:per_class_quota // 2]
            mid_k = sorted_items[len(sorted_items) // 2 : len(sorted_items) // 2 + per_class_quota // 2]

            selected += random.sample(top_k + mid_k, min(per_class_quota, len(top_k + mid_k)))

        # In case fewer than desired selected
        if len(selected) < num_selected:
            remaining = sorted(sample_scores, key=lambda x: x["entropy"], reverse=True)
            used_ids = {id(s["sample"]) for s in selected}
            for s in remaining:
                if id(s["sample"]) not in used_ids:
                    selected.append(s)
                if len(selected) == num_selected:
                    break

    else:
        # No class balancing, just global entropy diversity
        sorted_items = sorted(sample_scores, key=lambda x: x["entropy"], reverse=True)
        top_k = sorted_items[:num_selected // 2]
        mid_k = sorted_items[len(sorted_items) // 2 : len(sorted_items) // 2 + num_selected // 2]
        selected = random.sample(top_k + mid_k, num_selected)

    return [s["sample"] for s in selected]



# def prototype_based_sampling(sample_features, prototypes, num_selected=100, per_class=True, mem_size=1000):
#     """
#     Select diverse hard positive samples far from their class prototype.

#     Args:
#         sample_features: [(sample_dict, feature)]
#         prototypes: {label: prototype tensor}
#         num_selected: number of samples to return
#         per_class: select equally from each class
#         mem_size: effective memory size used to calculate sampling interval

#     Returns:
#         list of dict: selected samples
#     """
#     class_buckets = defaultdict(list)

#     # Compute distance from prototype
#     for sample, feat in sample_features:
#         proto = prototypes[sample['label']]
#         dist = 1 - F.cosine_similarity(feat.unsqueeze(0), proto.unsqueeze(0)).item()
#         class_buckets[sample['label']].append((dist, sample))

#     selected = []

#     def interval_sampling(items, num, mem_size):
#         # Sort by distance (hardest first)
#         items = sorted(items, key=lambda x: x[0], reverse=True)
#         interval = max(1, int(mem_size / num))
#         sampled = [items[i][1] for i in range(0, len(items), interval)][:num]
#         return sampled

#     if per_class:
#         per_class_quota = max(1, num_selected // len(class_buckets))
#         for cls, items in class_buckets.items():
#             selected += interval_sampling(items, min(per_class_quota, len(items)), mem_size)
#     else:
#         all_items = []
#         for items in class_buckets.values():
#             all_items.extend(items)
#         selected = interval_sampling(all_items, num_selected, mem_size)

#     return selected



def prototype_based_sampling(
    sample_features, 
    prototypes, 
    seen_classes, 
    num_selected, 
    distance_metric='cosine'  # or 'l2'
):
    """
    Randomly select class-balanced samples with remainder randomly distributed across classes.

    Args:
        sample_features: [(sample_dict, feature)]
        prototypes: {label: prototype tensor} (unused, kept for compatibility)
        seen_classes: number of seen classes so far
        num_selected: total number of samples to select

    Returns:
        list of dict: selected samples
    """
    class_buckets = defaultdict(list)

    # Group samples by class
    for sample, _ in sample_features:
        label = sample['label']
        class_buckets[label].append(sample)

    per_class_quota = num_selected // seen_classes  # e.g., 3
    remaining = num_selected % seen_classes         # e.g., 10

    selected = []

    # Step 1: Select per_class_quota samples per class
    eligible_classes = list(class_buckets.keys())
    for cls in eligible_classes:
        samples = class_buckets[cls]
        if len(samples) <= per_class_quota:
            selected += samples
        else:
            selected += random.sample(samples, per_class_quota)

    # Step 2: For remaining slots, pick extra samples from randomly chosen classes
    extra_classes = random.sample(eligible_classes, min(remaining, len(eligible_classes)))

    for cls in extra_classes:
        # Avoid selecting the same sample twice
        already_selected = set(id(s) for s in selected)
        candidates = [s for s in class_buckets[cls] if id(s) not in already_selected]
        if candidates:
            selected.append(random.choice(candidates))

    return selected




import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def extract_memory_features(
    model,
    memory_loader,
    device="cuda"
):
    """
    Extract features and class labels from memory loader
    """
    model.eval()
    features = []
    class_labels = []

    with torch.no_grad():
        for data in memory_loader:
            x = data["image"].to(device)
            y = data["label"]

            _,z = model.pcrForward(x)   # your forward
            # z = model.features(x)
            features.append(z.cpu())
            class_labels.append(y)

    features = torch.cat(features, dim=0)
    class_labels = torch.cat(class_labels, dim=0)

    return features, class_labels

def l2_normalize(features):
    features = features.numpy() if isinstance(features, torch.Tensor) else features
    return features / np.linalg.norm(features, axis=1, keepdims=True)

def plot_tsne_by_class(
    features,
    class_labels,
    title,
    save_path="./tsne_classes.png",
    dpi=300
):
    features = features.numpy() if isinstance(features, torch.Tensor) else features
    class_labels = class_labels.numpy() if isinstance(class_labels, torch.Tensor) else class_labels

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=0
    )
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=class_labels,
        cmap="tab20",
        s=6,
        alpha=0.7
    )

    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

def plot_tsne_subset_classes(
    features,
    class_labels,
    selected_classes,
    # title,
    save_path="./tsne_subset_classes.png",
    dpi=300
):
    features = features.numpy() if isinstance(features, torch.Tensor) else features
    class_labels = class_labels.numpy() if isinstance(class_labels, torch.Tensor) else class_labels

    mask = np.isin(class_labels, selected_classes)
    features = features[mask]
    class_labels = class_labels[mask]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=0
    )
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    for c in selected_classes:
        idx = class_labels == c
        plt.scatter(
            embeddings[idx, 0],
            embeddings[idx, 1],
            s=10,
            alpha=0.8,
            # label=f"Class {c}"
        )

    # plt.legend(markerscale=1.5)
    # plt.title(title)
    plt.axis("off")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


from collections import defaultdict

def group_by_class(samples):
    class_to_samples = defaultdict(list)
    for s in samples:
        class_to_samples[s["label"]].append(s)
    return class_to_samples

import numpy as np

def sample_dirichlet_proportions(num_classes, alpha, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.dirichlet([alpha] * num_classes)

import random

def induce_dirichlet_imbalance(samples, alpha=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    class_to_samples = group_by_class(samples)
    classes = sorted(class_to_samples.keys())

    proportions = sample_dirichlet_proportions(
        num_classes=len(classes),
        alpha=alpha
    )

    imbalanced_samples = []

    for cls, frac in zip(classes, proportions):
        cls_samples = class_to_samples[cls]
        random.shuffle(cls_samples)

        keep = max(1, int(len(cls_samples) * frac))
        imbalanced_samples.extend(cls_samples[:keep])

    return imbalanced_samples

