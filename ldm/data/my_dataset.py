# import os
# from glob import glob
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as T


# class MyDataset(Dataset):
#     def __init__(self, data_root):
#         super().__init__()
#         self.data_root = data_root
#         # print(self.data_root)
#         # print("检查 data_root:", self.data_root)

#         # print("data_root 真实存在吗？", os.path.exists(self.data_root))

#         # print("data_root 下内容:", os.listdir(self.data_root))


#         # 先拿到所有的 control_mean 路径
#         control_folders = [ d for d in os.listdir(data_root)
#                            if d.endswith("_control_mean") and os.path.isdir(os.path.join(data_root, d))]


#         self.data_pairs = []
#         # print(control_folders)

#         for folder in control_folders:
#             control_folder = os.path.join(data_root, folder)
#             target_folder = os.path.join(data_root, folder.replace("control_mean", "M0"))

#             # print(f"targeeeeeeeet:{target_folder}")

#             control_imgs = sorted(glob(os.path.join(control_folder, "*.png")))
#             target_imgs = sorted(glob(os.path.join(target_folder, "*.png")))

#             assert len(control_imgs) == len(target_imgs), f"{folder} 切片数量不一致"

#             for c_img, t_img in zip(control_imgs, target_imgs):
#                 self.data_pairs.append((c_img, t_img))

#         print(f"Found {len(self.data_pairs)} control-M0 slice pairs.")

#         self.transform = T.Compose([
#             T.Resize((256, 256)),
#             T.ToTensor(),
#             T.Normalize([0.5], [0.5])
#         ])

#     def __len__(self):
#         return len(self.data_pairs)

#     def __getitem__(self, idx):
#         control_path, target_path = self.data_pairs[idx]
#         control_img = self.load_image(control_path)  # [3, H, W]
#         target_img = self.load_image(target_path)    # [3, H, W]

#         return {
#             "image": target_img,
#             "control": control_img
#     }

import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        # print(self.data_root)
        # print("检查 data_root:", self.data_root)

        # print("data_root exist？", os.path.exists(self.data_root))

        # print("data_root 下内容:", os.listdir(self.data_root))


        # 先拿到所有的 control_mean 路径
        control_folders = [ d for d in os.listdir(data_root)
                           if d.endswith("_control_mean") and os.path.isdir(os.path.join(data_root, d))]


        self.data_pairs = []
        # print(control_folders)

        for folder in control_folders:
            control_folder = os.path.join(data_root, folder)
            target_folder = os.path.join(data_root, folder.replace("control_mean", "M0"))

            # print(f"targeeeeeeeet:{target_folder}")

            control_imgs = sorted(glob(os.path.join(control_folder, "*.png")))
            target_imgs = sorted(glob(os.path.join(target_folder, "*.png")))

            assert len(control_imgs) == len(target_imgs), f"{folder} 切片数量不一致"

            for c_img, t_img in zip(control_imgs, target_imgs):
                self.data_pairs.append((c_img, t_img))

        print(f"Found {len(self.data_pairs)} control-M0 slice pairs.")

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        control_path, target_path = self.data_pairs[idx]
        control = Image.open(control_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        control = self.transform(control)
        target = self.transform(target)

        
        # 拼接成 [6, H, W]
        combined = torch.cat((target, control), dim=0)


        return {"image": target, "cond": control}



