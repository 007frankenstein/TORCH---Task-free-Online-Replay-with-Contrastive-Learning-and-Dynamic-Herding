import torch
import random
import pandas as pd
from typing import Dict, List, Tuple

import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_loader import ImageDataset



# # # --- CeCR Prototype Normalization Function ---
@torch.no_grad()
def normalize_prototypes_(prototypes: Dict[int, torch.Tensor]):
    for c, p in prototypes.items():
        if p is None:
            continue
        prototypes[c] = p / (p.norm(p=2) + 1e-8)


# # # --- CeCR prototype Update Function ---
@torch.no_grad()
def prototype_update(
    features: torch.Tensor,   # shape: [B, D]
    labels: torch.Tensor,     # shape: [B]
    prototypes: dict,
    alpha: float = 0.9,
):
    """
    Updates prototypes using Exponential Moving Average (EMA).
    """
    # Ensure labels are on the correct device
    labels = labels.to(features.device)
    
    # Get unique classes in the current batch
    classes_in_batch = labels.unique().tolist()

    for c in classes_in_batch:
        # Create a mask for class c
        mask = (labels == c)
        
        # Extract features for this class
        Bc = features[mask]  # [N_c, D]
        
        # Calculate current batch mean
        p_bar = Bc.mean(dim=0).detach()
        
        # Normalize the mean update to unit sphere
        p_bar = F.normalize(p_bar, p=2, dim=0, eps=1e-8)

        if c not in prototypes:
            # First observation: Initialize prototype
            prototypes[c] = p_bar
        else:
            # Update existing prototype
            old_proto = prototypes[c]
            new_proto = alpha * old_proto + (1.0 - alpha) * p_bar
            
            # Re-normalize after update to maintain unit sphere constraint
            prototypes[c] = F.normalize(new_proto, p=2, dim=0, eps=1e-8)



# # # --- CeCR Memory Update Function ---
@torch.no_grad()
def memory_update(
    logger,
    Bn: list,                    # current batch metadata (list of dicts)
    M: dict,                     # Memory dict {label: [samples]}
    Enc,                         # Encoder model
    device,
    prototypes: dict,
    Limit: int,
    max_i: int,
    dataset_name: str,
    transform,
):
    """
    Updates the replay memory using the CeCR/NCM strategy.
    Efficiently manages memory size and optimizes sample selection.
    """
    Enc.eval()

    # Step 1–3: Reduce existing replay sets if Limit has decreased
    for c in list(M.keys()):
        M[c] = reduce_replay_set(M[c], Limit)

    # Step 4: Process current batch
    for sample in Bn:
        y_j = sample["label"]

        # Step 5–7: Initialize replay set if it doesn't exist
        if y_j not in M:
            M[y_j] = [sample]
            continue

        # Step 8–11: Fill replay set if not full
        if len(M[y_j]) < Limit:
            M[y_j].append(sample)
            continue

        # -------------------------------------------------------
        # Algorithm 2: Pseudo-Test & Optimization (Steps 12-23)
        # -------------------------------------------------------
        
        # Identify which samples in memory are currently "wrong" (misclassified)
        V_right, V_wrong = pseudo_test(
            M[y_j], Enc, prototypes, dataset_name, transform, device
        )

        # Temporary list combining all samples
        M[y_j] = V_right + V_wrong

        # Optimization loop: Try to replace wrong samples with the new sample
        # to see if it improves the Class Mean's proximity to the Prototype.
        for _ in range(max_i):
            if not V_wrong:
                break

            # Pick a random "wrong" sample to potentially evict
            x_h = random.choice(V_wrong)

            # Compute current class mean (mu_o)
            mu_o = compute_class_mean(M[y_j], Enc, device, dataset_name, transform)

            # Create candidate set: Remove x_h, add new sample
            M_candidate = [x for x in M[y_j] if x is not x_h]
            M_candidate.append(sample)

            # Compute new class mean (mu_r)
            mu_r = compute_class_mean(M_candidate, Enc, device, dataset_name, transform)

            # Compare distances to the prototype
            dist_old = torch.norm(mu_o - prototypes[y_j], p=2)
            dist_new = torch.norm(mu_r - prototypes[y_j], p=2)

            # If new mean is closer to prototype, keep the change
            if dist_new < dist_old:
                M[y_j] = M_candidate
                break
                
    return M

def reduce_replay_set(memory: list, limit: int):
    if len(memory) <= limit:
        return memory
    return memory[:limit]


@torch.no_grad()
def pseudo_test(
    memory: list,
    Enc,
    prototypes: dict,
    dataset_name: str,
    transform,
    device="cuda:0"
):
    """
    Classifies memory samples using NCM (Nearest Class Mean).
    Uses Vectorized Batch Processing for speed.
    """
    Enc.eval()

    if not memory:
        return [], []
    if not prototypes:
        return memory, []

    # 1. Prepare Prototypes Matrix
    # We sort keys to ensure index consistency
    proto_keys = sorted(list(prototypes.keys()))
    proto_tensor = torch.stack([prototypes[k] for k in proto_keys]).to(device)
    proto_tensor = F.normalize(proto_tensor, p=2, dim=1, eps=1e-8) # [C, D]

    # 2. Vectorized Feature Extraction
    mem_dataset = ImageDataset(
        pd.DataFrame(memory),
        dataset=dataset_name,
        transform=transform,
    )
    # num_workers=0 is crucial here to avoid spawning overhead for small batches
    mem_loader = DataLoader(mem_dataset, batch_size=32, shuffle=False, num_workers=0) 
    
    all_z = []
    for batch in mem_loader:
        x = batch["image"].to(device)
        z = Enc.cecr_forward(x)
        z = F.normalize(z, p=2, dim=1, eps=1e-8)
        all_z.append(z)
        
    all_z = torch.cat(all_z, dim=0) # [N, D]

    # 3. NCM Classification via Matrix Multiplication
    # Similarity = z @ proto.T. Max sim = Min distance (for unit vectors)
    sims = torch.mm(all_z, proto_tensor.T) # [N, C]
    pred_indices = sims.argmax(dim=1).cpu().tolist() # [N]
    
    V_right, V_wrong = [], []
    
    for i, sample in enumerate(memory):
        pred_label = proto_keys[pred_indices[i]]
        if pred_label == sample["label"]:
            V_right.append(sample)
        else:
            V_wrong.append(sample)

    return V_right, V_wrong


@torch.no_grad()
def compute_class_mean(
    memory: list,
    Enc,
    device,
    dataset_name: str,
    transform,
):
    """
    Computes the mean feature vector of a memory set.
    Uses Vectorized Batch Processing.
    """
    Enc.eval()
    if not memory:
        raise ValueError("Empty replay set")

    # 1. Vectorized Load
    mem_dataset = ImageDataset(
        pd.DataFrame(memory),
        dataset=dataset_name,
        transform=transform,
    )
    # Batch size can be len(memory) since replay sets are small (e.g., 20)
    mem_loader = DataLoader(mem_dataset, batch_size=64, shuffle=False, num_workers=0)

    features = []
    for batch in mem_loader:
        x = batch["image"].to(device)
        z = Enc.cecr_forward(x)
        z = F.normalize(z, p=2, dim=1, eps=1e-8)
        features.append(z)

    features = torch.cat(features, dim=0)
    
    # 2. Compute Mean
    mean_vector = features.mean(dim=0)
    
    # Note: We return the raw mean. NCM usually uses the raw mean of normalized features.
    return mean_vector



# # # --- CeCR Memory Retrieval Function ---
def memory_retrieval(
    M: dict,
    C: set,
    C_m: set,
    n_retrieve: int,
):
    """
    Algorithm 1: Memory Retrieval using Selection Without Replacement (SWR).
    """
    B_m = []

    # Filter for classes that actually have data in memory
    valid_classes = {c for c in C if c in M and len(M[c]) > 0}

    if not valid_classes:
        return [], C_m

    # Identify classes we haven't retrieved from recently (SWR logic)
    C_d = list(valid_classes - C_m)

    # If we have exhausted all classes, reset the tracking set
    if not C_d:
        C_m.clear()
        C_d = list(valid_classes)

    # Determine how many classes to sample
    k = min(n_retrieve, len(C_d))
    
    # Randomly select k classes
    C_r = random.sample(C_d, k)

    # Sample one image per selected class
    for c_r in C_r:
        if M[c_r]: # Safety check
            B_m.append(random.choice(M[c_r]))

    # Update the set of retrieved classes
    C_m.update(C_r)

    return B_m, C_m