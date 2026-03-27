import os
from torch.utils.data import Dataset
import cv2


class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="train.txt",
            val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

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

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "idx": idx}
        return sample
