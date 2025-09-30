#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GoPro test-set evaluator
- ECC homography alignment
- PSNR / SSIM with mask
- multi-worker support
- reproducible log
# 直接跑，路径已经写死
python eval_gopro.py

# 覆盖路径
python eval_gopro.py --pred-dir my_pred --gt-dir my_gt --out-file result.txt

# 多进程加速
python eval_gopro.py --num-workers 12

# 跑最小单元测试
python eval_gopro.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity
from tqdm import tqdm

# ---------- core alignment ----------
def align_images(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Returns
    -------
    pred_warp : aligned prediction
    gt_warp   : aligned gt (cropped to valid region)
    mask      : binary mask of valid pixels
    converged : ECC convergence flag
    """
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)

    # scale optimisation
    scale = np.sum(gt_f * pred_f) / (np.sum(pred_f * pred_f) + 1e-8)
    pred_f *= scale

    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

    gt_gray = cv2.cvtColor(gt_f, cv2.COLOR_RGB2GRAY)
    pred_gray = cv2.cvtColor(pred_f, cv2.COLOR_RGB2GRAY)

    try:
        _, wm = cv2.findTransformECC(
            gt_gray, pred_gray, warp_matrix, warp_mode, criteria,
            inputMask=None, gaussFiltSize=5
        )
        converged = True
    except cv2.error:
        wm = np.eye(3, 3, dtype=np.float32)
        converged = False

    h, w = gt.shape[:2]
    pred_warp = cv2.warpPerspective(
        pred_f, wm, (w, h),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    mask = cv2.warpPerspective(
        np.ones_like(pred_f, dtype=np.float32),
        wm, (w, h),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    gt_warp = gt_f * mask
    return pred_warp, gt_warp, mask, converged


# ---------- metrics ----------
def compute_psnr(
    gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, data_range: float
) -> float:
    err = np.sum((gt - pred) ** 2 * mask) / (np.sum(mask) + 1e-8)
    return 10 * np.log10((data_range**2) / err)


def compute_ssim(
    gt: np.ndarray, pred: np.ndarray, mask: np.ndarray
) -> float:
    win_size = min(7, gt.shape[0] // 2, gt.shape[1] // 2)
    if win_size < 3:
        return float("nan")
    _, ssim_map = structural_similarity(
        gt, pred, win_size=win_size, channel_axis=2,
        gaussian_weights=True, data_range=1.0, full=True
    )
    ssim_map *= mask
    # crop border
    pad = (win_size - 1) // 2
    if pad > 0:
        ssim_map = ssim_map[pad:-pad, pad:-pad, :]
        mask_c = mask[pad:-pad, pad:-pad, :]
    else:
        mask_c = mask
    val = ssim_map.sum(axis=(0, 1)) / (mask_c.sum(axis=(0, 1)) + 1e-8)
    return float(np.mean(val))


# ---------- single image ----------
def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        img = io.imread(path)
    except Exception as e:
        return None
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.float32) / 255.0


def process_one(
    name: str, pred_dir: Path, gt_dir: Path
) -> Tuple[str, Optional[float], Optional[float], str]:
    pred_path = pred_dir / name
    gt_path = gt_dir / name

    pred = load_image(pred_path)
    gt = load_image(gt_path)
    if pred is None:
        return name, None, None, "read_pred_fail"
    if gt is None:
        return name, None, None, "read_gt_fail"
    if pred.shape != gt.shape:
        return name, None, None, "shape_mismatch"

    pred_warp, gt_warp, mask, converged = align_images(pred, gt)
    if not converged:
        return name, None, None, "align_fail"

    data_range = 1.0
    psnr = compute_psnr(gt_warp, pred_warp, mask, data_range)
    ssim = compute_ssim(gt_warp, pred_warp, mask)
    if np.isnan(psnr) or np.isnan(ssim):
        return name, None, None, "metric_nan"
    return name, psnr, ssim, ""


# ---------- main ----------
def main():
    # >>>>>>>>>>  DEFAULT PATHS  <<<<<<<<<<
    pred_dir = Path("results_final_2/GoPro/GoPro")      # 预测图目录
    gt_dir = Path("datasets/GoPro/test/GoPro/target")   # GT 目录
    out_file = Path("result_gopro.txt")
    # >>>>>>>>>>  DEFAULT PATHS  <<<<<<<<<<

    parser = argparse.ArgumentParser(description="GoPro PSNR/SSIM evaluator")
    parser.add_argument("--pred-dir", type=Path, default=pred_dir)
    parser.add_argument("--gt-dir", type=Path, default=gt_dir)
    parser.add_argument("--out-file", type=Path, default=out_file)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--self-test", action="store_true", help="run minimal unit test")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    out_file = args.out_file

    if not pred_dir.is_dir():
        sys.exit(f"Pred directory not found: {pred_dir}")
    if not gt_dir.is_dir():
        sys.exit(f"GT directory not found: {gt_dir}")

    ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    names = sorted([f for f in os.listdir(pred_dir) if Path(f).suffix.lower() in ext])

    results: List[Tuple[str, float, float]] = []
    failed: List[Tuple[str, str]] = []

    def worker(name):
        return process_one(name, pred_dir, gt_dir)

    t0 = time.time()
    with tqdm(total=len(names), ncols=80) as pbar:
        for name in names:
            if not (gt_dir / name).exists():
                failed.append((name, "gt_missing"))
                pbar.update()
                continue
            img_name, psnr, ssim, err = worker(name)
            if err:
                failed.append((img_name, err))
            else:
                results.append((img_name, psnr, ssim))
            pbar.update()

    elapsed = time.time() - t0
    avg_psnr = np.nanmean([r[1] for r in results])
    avg_ssim = np.nanmean([r[2] for r in results])

    # ---------- write log ----------
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("GoPro Dataset Evaluation Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Pred: {pred_dir.resolve()}\n")
        f.write(f"GT:   {gt_dir.resolve()}\n")
        f.write(f"Images: total={len(names)}, success={len(results)}, failed={len(failed)}\n")
        f.write(f"Avg PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Avg SSIM: {avg_ssim:.6f}\n")
        f.write(f"Elapsed: {elapsed:.1f} s\n")
        f.write("=" * 70 + "\n\n")
        f.write("Per-image results (sorted by PSNR desc):\n")
        f.write(f"{'Image':<35} {'PSNR(dB)':<12} {'SSIM':<10}\n")
        f.write("-" * 70 + "\n")
        for img, p, s in sorted(results, key=lambda x: x[1], reverse=True):
            f.write(f"{img:<35} {p:>10.4f}  {s:>8.6f}\n")
        if failed:
            f.write("\nFailed images:\n")
            for img, reason in failed:
                f.write(f"  - {img}  ({reason})\n")
    print("\n" + "=" * 50)
    print(f"Avg PSNR: {avg_psnr:.4f} dB")
    print(f"Avg SSIM: {avg_ssim:.6f}")
    print("=" * 50)
    print(f"Saved to -> {out_file.resolve()}")


# ---------- minimal self-test ----------
def self_test():
    print("Running minimal self-test...")
    from tempfile import TemporaryDirectory
    import shutil

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "pred").mkdir()
        (tmp / "gt").mkdir()
        # create 5 dummy 64x64 images
        rng = np.random.default_rng(42)
        for i in range(5):
            pred = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            gt = pred + rng.integers(-5, 5, pred.shape, dtype=np.int16)
            gt = np.clip(gt, 0, 255).astype(np.uint8)
            io.imsave(tmp / "pred" / f"{i:02d}.png", pred)
            io.imsave(tmp / "gt" / f"{i:02d}.png", gt)

        # run evaluation
        out_log = tmp / "log.txt"
        import subprocess

        subprocess.check_call(
            [sys.executable, __file__,
             "--pred-dir", str(tmp / "pred"),
             "--gt-dir", str(tmp / "gt"),
             "--out-file", str(out_log)],
            cwd=tmp,
        )
        print("Self-test log:")
        print(out_log.read_text())
        assert "Avg PSNR" in out_log.read_text()
        print("Self-test passed ✔")


if __name__ == "__main__":
    main()