import os
from PIL import Image, ImageFile
from torchvision.datasets.folder import has_file_allowed_extension
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def make_dataset_with_ids(dir, class_to_idx, extensions):
    """
    Generates a list of samples of a form (path, class_index, patient_id, slice_number).
    """
    samples = []
    dir = os.path.expanduser(dir)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(dir, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    patient_id, slice_number = fname.split('_')[:2]
                    samples.append((path, class_index, patient_id, slice_number))
    return samples

def align_samples(view_samples):
    """
    Aligns samples from multiple views by patient_id and slice_number.
    """
    aligned_samples = []
    num_views = len(view_samples)
    sample_dict = {}
    
    for view in view_samples:
        for sample in view:
            key = (sample[2], sample[3])  # (patient_id, slice_number)
            if key not in sample_dict:
                sample_dict[key] = [None] * num_views
            sample_dict[key][view_samples.index(view)] = sample
    
    for key, samples in sample_dict.items():
        if all(samples):
            aligned_samples.append(samples)
    
    return aligned_samples

class MultiViewDataset(Dataset):
    def __init__(self, data_dirs, view_transforms=None, labeled=True, extensions=('png', 'jpg', 'jpeg')):
        """
        Args:
            data_dirs (list of strings): List of directories for each view.
            view_transforms (list of callables, optional): List of transforms for each view.
            labeled (bool): Whether the dataset is labeled or not.
            extensions (tuple): Allowed image extensions.
        """
        assert len(data_dirs) == len(view_transforms), "Each view must have a corresponding transform"
        self.view_transforms = view_transforms
        self.data_dirs = data_dirs
        self.labeled = labeled

        self.classes, self.class_to_idx = find_classes(self.data_dirs[0])
        self.int_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Create datasets for each view with patient_id and slice_number
        self.view_samples = [make_dataset_with_ids(data_dir, self.class_to_idx, extensions) for data_dir in self.data_dirs]

        # Align samples from multiple views
        self.aligned_samples = align_samples(self.view_samples)

    def __len__(self):
        return len(self.aligned_samples)

    def __getitem__(self, index):
        sample = {}
        aligned_sample = self.aligned_samples[index]

        if self.labeled:
            sample['label'] = aligned_sample[0][1]

        for i, (view_path, class_index, patient_id, slice_number) in enumerate(aligned_sample):
            view_image = Image.open(view_path).convert('RGB')
            if self.view_transforms[i]:
                view_image = self.view_transforms[i](view_image)
            sample[f'v{i+1}'] = view_image
        
        return sample
