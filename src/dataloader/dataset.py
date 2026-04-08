import os
from torch.utils.data import Dataset
import cv2
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None


def build_signed_distance_target(mask, max_distance=8.0):
    if distance_transform_edt is None:
        raise ImportError("DistanceAux data preparation requires scipy.")

    foreground = mask[..., 0] > 0.5
    target = np.zeros_like(mask, dtype=np.float32)
    if not foreground.any():
        return target

    pos_dist = distance_transform_edt(foreground)
    neg_dist = distance_transform_edt(~foreground)
    signed_distance = (neg_dist - pos_dist) / float(max_distance)
    target[..., 0] = np.clip(signed_distance, -1.0, 1.0)
    return target


class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="train.txt",
            val_file_dir="val.txt",
            use_distance_aux=False,
            distance_max=8.0,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []
        self.use_distance_aux = use_distance_aux
        self.distance_max = distance_max

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        # 自动查找图像文件的扩展名 (png, jpg, jpeg)
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            current_path = os.path.join(self._base_dir, 'images', case + ext)
            if os.path.exists(current_path):
                image_path = current_path
                break

        if image_path is None:
            raise FileNotFoundError(
                f"Image file not found for case '{case}' in {os.path.join(self._base_dir, 'images')}. "
                "Tried .png, .jpg, and .jpeg.")

        # 自动查找掩码文件的扩展名 (png, jpg, jpeg)
        # 保持原始代码中 'masks/0' 的路径结构
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            current_path = os.path.join(self._base_dir, 'masks', '0', case + ext)
            if os.path.exists(current_path):
                mask_path = current_path
                break

        if mask_path is None:
            raise FileNotFoundError(
                f"Mask file not found for case '{case}' in {os.path.join(self._base_dir, 'masks', '0')}. "
                "Tried .png, .jpg, and .jpeg.")

        # 使用找到的路径加载图像和掩码
        image = cv2.imread(image_path)
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32')
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        distance_target = None
        if self.use_distance_aux:
            distance_target = build_signed_distance_target(label, max_distance=self.distance_max)
        label = label.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "idx": idx}
        if distance_target is not None:
            sample["distance_target"] = distance_target.transpose(2, 0, 1)
        return sample
