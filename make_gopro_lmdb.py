import os
from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs


def build_lmdb(folder, lmdb_out, suffix=("png", "jpg", "jpeg"), n_thread=8):
    names = sorted(list(scandir(folder, suffix=suffix, recursive=False)))
    keys = [os.path.splitext(n)[0] for n in names]  # 去掉扩展名作为 key
    make_lmdb_from_imgs(
        data_path=folder,
        lmdb_path=lmdb_out,
        img_path_list=names,  # 基于 folder 的相对文件名
        keys=keys,
        multiprocessing_read=True,  # 机器内存足够时可提速；不足就设 False
        n_thread=n_thread,
    )


# 训练集
build_lmdb("datasets/GoPro/train/input", "datasets/GoPro/train/blur_crops.lmdb")
build_lmdb("datasets/GoPro/train/target", "datasets/GoPro/train/sharp_crops.lmdb")

# 测试集（验证）
build_lmdb(
    "datasets/GoPro/test/GoPro/input", "datasets/GoPro/test/GoPro/blur_crops.lmdb"
)
build_lmdb(
    "datasets/GoPro/test/GoPro/target", "datasets/GoPro/test/GoPro/sharp_crops.lmdb"
)
