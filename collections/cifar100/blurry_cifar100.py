import numpy as np
import json
import os

class BlurryCIFAR100Simulation:
    def __init__(self, num_tasks=10, n_blurry=50, m_blurry=10, train_samples_per_class=500, root_dir='./datasets/cifar100'):
        self.num_tasks = num_tasks
        self.n_blurry = n_blurry  # Number of blurry classes
        self.m_blurry = m_blurry  # Samples per blurry class for non-head classes
        self.train_samples_per_class = train_samples_per_class
        self.head_class_samples = train_samples_per_class - (num_tasks - 1) * m_blurry  # 500 - 9 * 10 = 410
        self.root_dir = root_dir  # Path to the dataset root directory
        
        # Mapping class names to class indices
        self.class_names = sorted(os.listdir(os.path.join(self.root_dir, 'train')))  # List of class names (e.g., 'airplane', 'tree', etc.)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}  # Reverse mapping for debugging

        # Map filenames to classes for both train and test
        self.class_to_train_filenames = self._map_filenames_to_classes(split='train')
        self.class_to_test_filenames = self._map_filenames_to_classes(split='test')

        # Split classes into disjoint and blurry
        self.disjoint_classes = np.arange(self.num_tasks * 5)  # First 50 classes for disjoint part
        self.blurry_classes = np.arange(self.num_tasks * 5, len(self.class_names))  # Last 50 classes for blurry part

        # Generate task splits
        self.train_task_indices = []
        self.test_task_indices = []
        self._create_task_splits()

    def _map_filenames_to_classes(self, split='train'):
        """
        Map image filenames in the dataset directory to their corresponding class for the given split ('train' or 'test').
        Assumes that the directory structure is like datasets/cifar100/split/class_name/filename.png
        """
        class_to_filenames = {}
        split_dir = os.path.join(self.root_dir, split)
        
        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                filenames = [os.path.join(class_name, f) for f in os.listdir(class_dir) if f.endswith('.png')]
                class_to_filenames[self.class_to_idx[class_name]] = filenames
        return class_to_filenames

    def _create_task_splits(self):
        for t in range(self.num_tasks):
            train_indices = []
            test_indices = []

            # Disjoint classes for the current task (5 new disjoint classes per task)
            current_disjoint_classes = self.disjoint_classes[t * 5:(t + 1) * 5]
            for cls in current_disjoint_classes:
                if cls in self.class_to_train_filenames:
                    train_indices.extend(self.class_to_train_filenames[cls])  # Use all samples for disjoint class
                else:
                    print(f"Warning: Class {cls} (mapped to {self.idx_to_class[cls]}) not found in training directory structure")

            # Blurry classes for the current task
            current_head_classes = self.blurry_classes[t * 5:(t + 1) * 5]  # 5 head classes for the current task
            current_blurry_minor_classes = np.setdiff1d(self.blurry_classes, current_head_classes)  # Remaining 45 blurry classes

            # For head classes: Assign 410 samples
            for cls in current_head_classes:
                if cls in self.class_to_train_filenames:
                    train_indices.extend(self.class_to_train_filenames[cls][:self.head_class_samples])
                else:
                    print(f"Warning: Head class {cls} (mapped to {self.idx_to_class[cls]}) not found in training directory structure")

            # For other blurry classes: Assign 10 samples each
            for cls in current_blurry_minor_classes:
                if cls in self.class_to_train_filenames:
                    train_indices.extend(self.class_to_train_filenames[cls][:self.m_blurry])
                else:
                    print(f"Warning: Blurry class {cls} (mapped to {self.idx_to_class[cls]}) not found in training directory structure")

            # Record train indices for this task
            self.train_task_indices.append(train_indices)

            # Test indices include all samples for all seen disjoint and blurry classes up to this task
            seen_disjoint_classes = self.disjoint_classes[:(t + 1) * 5]  # All disjoint classes seen so far
            seen_blurry_classes = self.blurry_classes  # All blurry classes are always included

            # Add all test samples from seen disjoint classes
            for cls in seen_disjoint_classes:
                if cls in self.class_to_test_filenames:
                    test_indices.extend(self.class_to_test_filenames[cls])  # Use all samples for disjoint classes
                else:
                    print(f"Warning: Disjoint class {cls} (mapped to {self.idx_to_class[cls]}) not found in test directory structure")

            # Add all test samples from all blurry classes (seen and unseen)
            for cls in seen_blurry_classes:
                if cls in self.class_to_test_filenames:
                    test_indices.extend(self.class_to_test_filenames[cls])  # Use all samples for blurry classes
                else:
                    print(f"Warning: Blurry class {cls} (mapped to {self.idx_to_class[cls]}) not found in test directory structure")

            # Record test indices for this task
            self.test_task_indices.append(test_indices)

            # Save train and test data information for the current task
            self.save_json(t, train_indices, test_indices)

    def save_json(self, task_id, train_indices, test_indices):
        # Prepare data information for training
        train_data_info = []
        for filename in train_indices:
            class_name = os.path.dirname(filename)  # Extract class name from file path
            full_filename = os.path.join(self.root_dir, 'train', filename)
            class_idx = self.class_to_idx[class_name]
            train_data_info.append({
                "klass": class_name,
                "filename": "train/" + filename,
                "label": class_idx  # Use class index as label
            })

        # Save training data information to JSON
        with open(f'cifar100_train_blurry10_rand1_cls55_task_{task_id}.json', 'w') as f:
            json.dump(train_data_info, f, indent=4)

        # Prepare data information for testing
        test_data_info = []
        for filename in test_indices:
            class_name = os.path.dirname(filename)  # Extract class name from file path
            full_filename = os.path.join(self.root_dir, 'test', filename)
            class_idx = self.class_to_idx[class_name]
            test_data_info.append({
                "klass": class_name,
                "filename": "test/" + filename,
                "label": class_idx  # Use class index as label
            })

        # Save testing data information to JSON
        with open(f'cifar100_test_blurry10_rand1_cls55_task_{task_id}.json', 'w') as f:
            json.dump(test_data_info, f, indent=4)

    def get_train_samples_per_class_per_task(self):
        """
        For each task, print the number of training samples for each class.
        """
        for t in range(self.num_tasks):
            class_count = {}
            for filename in self.train_task_indices[t]:
                class_name = os.path.dirname(filename)
                class_idx = self.class_to_idx[class_name]
                class_count[class_idx] = class_count.get(class_idx, 0) + 1
            print(f"Task {t + 1} - Training Samples Per Class:")
            for cls, count in sorted(class_count.items()):
                print(f"Class {cls}: {count} samples")
            print('-' * 50)

# Example of usage
if __name__ == "__main__":
    num_tasks = 10  # Define number of tasks
    n_blurry = 50   # Number of blurry classes
    m_blurry = 10   # Number of samples per blurry class for minor classes
    root_dir = '/u/student/2022/cs22resch01004/continual-learning-implemented/clai/dataset/cifar100'  # Path to the dataset directory

    dataset = BlurryCIFAR100Simulation(num_tasks=num_tasks, n_blurry=n_blurry, m_blurry=m_blurry, root_dir=root_dir)
    
    # To print number of train samples per class per task
    dataset.get_train_samples_per_class_per_task()
