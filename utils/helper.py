import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

def compute_and_plot_task_confusion_matrix(
    model,
    classifier,
    minibatch_dataset,
    num_tasks,
    batch_size=128,
    device="cuda",
    title="Task-Level Confusion Matrix",
    save_path=None
):
    """
    Computes and plots the task-level confusion matrix (e.g., for CIFAR-100 or CIFAR-10 split).

    Parameters:
        model: Trained model in eval mode.
        minibatch_dataset: Dataset object (ImageDataset with keys 'image' and 'label').
        num_tasks: Number of tasks (e.g., 10 for CIFAR-100, 5 for CIFAR-10).
        batch_size: Batch size for DataLoader.
        device: Device string ("cuda" or "cpu").
        title: Title for the plot.
        save_path: If given, saves the plot to this path (e.g., "conf_matrix.png").

    Returns:
        conf_matrix: NumPy array of shape [num_tasks, num_tasks]
    """
    model.eval()
    classifier.eval()
    # loader = DataLoader(minibatch_dataset, batch_size=batch_size, shuffle=False)

    # Determine total number of classes from one forward pass
    with torch.no_grad():
        for batch in minibatch_dataset:
            inputs = batch["image"].to(device)
            outputs = classifier(model(inputs))
            total_classes = outputs.shape[1]
            break
        print(total_classes)
    classes_per_task = total_classes // num_tasks
    conf_matrix = np.zeros((num_tasks, num_tasks), dtype=int)

    with torch.no_grad():
        for batch in minibatch_dataset:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = classifier(model(inputs))
            _, preds = torch.max(outputs, dim=1)

            true_tasks = (labels // classes_per_task).cpu().numpy()
            pred_tasks = (preds // classes_per_task).cpu().numpy()

            true_tasks = np.clip(true_tasks, 0, num_tasks - 1)
            pred_tasks = np.clip(pred_tasks, 0, num_tasks - 1)

            for t_true, t_pred in zip(true_tasks, pred_tasks):
                conf_matrix[t_true, t_pred] += 1

    # Plotting
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        conf_matrix,
        annot=False,
        cmap="YlGnBu",
        xticklabels=[str(i) for i in range(num_tasks)],
        yticklabels=[str(i) for i in range(num_tasks)],
        cbar=True
    )
    ax.set_xlabel("Prediction Task Label")
    ax.set_ylabel("True Task Label")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
    return conf_matrix


