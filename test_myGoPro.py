# 测试GoPro: python test_myGoPro.py
# 本脚本仅推理图像不计算指标
import os
import torch
import argparse
from models.EVSSM import EVSSM
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Image
from tqdm import tqdm


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, "input/"))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, "input", self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, "target", self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label
    @staticmethod
    def _check_image(lst):
        allowed = {"png", "jpg", "jpeg"}
        # 原地保留合法文件
        lst[:] = [f for f in lst
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # @staticmethod
    # def _check_image(lst):
    #     for x in lst:
    #         splits = x.split(".")
    #         if splits[-1] not in ["png", "jpg", "jpeg"]:
    #             raise ValueError


def test_dataloader(path, batch_size=1, num_workers=0):
    # 修改：根据新的数据集路径结构调整
    image_dir = os.path.join(
        path, "test/GoPro"
    )  # 原：image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def main(args):
    # CUDNN
    # cudnn.benchmark = True
    #
    if not os.path.exists("results/" + args.model_name + "/"):
        os.makedirs("results/" + args.model_name + "/")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = EVSSM()
    # print(model)
    if torch.cuda.is_available():
        model.cuda()

    _eval(model, args)


def _eval(model, args):
    state_dict = torch.load(args.test_model)["params"]
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda")
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()

    model.eval()

    with torch.no_grad():
        # 增强：为进度条添加总数信息和更详细的描述
        print(f"开始测试，共 {len(dataloader)} 张图片")

        # Main Evaluation
        # 修改：增强进度条显示，添加描述信息和进度统计
        for iter_idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="测试进度", unit="张"
        ):
            # 原：for iter_idx, data in tqdm(enumerate(dataloader)):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            b, c, h, w = input_img.shape
            #            h_n = (4 - h % 4) % 4
            #            w_n = (4 - w % 4) % 4
            #            input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')

            pred = model(input_img)
            torch.cuda.synchronize()
            #            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)

            save_name = os.path.join(args.result_dir, name[0])
            pred_clip += 0.5 / 255
            pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
            pred.save(save_name)

        # 增强：添加完成提示
        print(f"测试完成！结果已保存到：{args.result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--model_name", default="GoPro", type=str)
    # 修改：更新数据集路径
    parser.add_argument("--data_dir", type=str, default="datasets/GoPro/")
    

    # Test
    # 修改：更新模型权重路径
    parser.add_argument("--test_model", type=str, default="checkpoints/net_g_GoPro.pth")
    
    parser.add_argument("--save_image", type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.result_dir = os.path.join("results_final_2/", args.model_name, "GoPro/")
    print(args)
    main(args)
