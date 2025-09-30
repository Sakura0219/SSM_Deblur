import os
import cv2
from basicsr.utils import scandir
from basicsr.utils.lmdb_util import LmdbMaker


def list_images_map(folder, suffix=("png", "jpg", "jpeg"), recursive=False):
    """Return a dict: {stem: relative_path} for images under folder."""
    names = sorted(list(scandir(folder, suffix=suffix, recursive=recursive)))
    return {os.path.splitext(n)[0]: n for n in names}


def read_and_encode(path, compress_level=1):
    """Read image and encode to PNG bytes. Return (img_byte, (h,w,c)) or None on failure."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    ok, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    if not ok:
        return None
    return img_byte, (h, w, c)


def build_paired_lmdb(inp_dir, tgt_dir, out_inp_lmdb, out_tgt_lmdb, suffix=("png", "jpg", "jpeg"), recursive=False):
    """Build two LMDBs (blur/sharp) with exactly matched keys.

    Steps:
    - enumerate both sides, pair by filename stem
    - optionally drop unreadable/corrupted pairs
    - write both LMDBs in lockstep so keys stay aligned
    """
    inp_map = list_images_map(inp_dir, suffix=suffix, recursive=recursive)
    tgt_map = list_images_map(tgt_dir, suffix=suffix, recursive=recursive)

    common = sorted(set(inp_map.keys()) & set(tgt_map.keys()))
    if len(common) == 0:
        print(f"[SKIP] No common keys between {inp_dir} and {tgt_dir}")
        return False

    print(f"Create paired lmdb for {inp_dir} + {tgt_dir}\n  -> {out_inp_lmdb}\n  -> {out_tgt_lmdb}")
    maker_inp = LmdbMaker(out_inp_lmdb)
    maker_tgt = LmdbMaker(out_tgt_lmdb)

    kept, dropped = 0, 0
    for k in common:
        inp_path = os.path.join(inp_dir, inp_map[k])
        tgt_path = os.path.join(tgt_dir, tgt_map[k])
        inp_obj = read_and_encode(inp_path)
        tgt_obj = read_and_encode(tgt_path)
        if inp_obj is None or tgt_obj is None:
            print(f"[DROP] Unreadable pair key={k} (inp_ok={inp_obj is not None}, tgt_ok={tgt_obj is not None})")
            dropped += 1
            continue
        inp_byte, inp_shape = inp_obj
        tgt_byte, tgt_shape = tgt_obj
        maker_inp.put(inp_byte, k, inp_shape)
        maker_tgt.put(tgt_byte, k, tgt_shape)
        kept += 1
        if kept % 500 == 0:
            print(f"  progress: kept={kept}, dropped={dropped}")

    maker_inp.close()
    maker_tgt.close()
    print(f"[OK] Wrote paired lmdbs. kept={kept}, dropped={dropped}")
    if kept == 0:
        print("[WARN] No valid pairs written.")
        return False
    return True


def build_split(root, recursive=False):
    inp = os.path.join(root, "input")
    tgt = os.path.join(root, "target")
    out_inp = os.path.join(root, "blur.lmdb")
    out_tgt = os.path.join(root, "sharp.lmdb")
    ok = build_paired_lmdb(inp, tgt, out_inp, out_tgt, recursive=recursive)
    if not ok:
        print(f"[WARN] Split not completed for {root}")


def build_realblur_tree(base_dir="datasets/RealBlur", recursive=False):
    for phase in ("train", "test"):
        phase_dir = os.path.join(base_dir, phase)
        if not os.path.isdir(phase_dir):
            print(f"[SKIP] Phase dir not found: {phase_dir}")
            continue
        for subset in sorted(os.listdir(phase_dir)):
            root = os.path.join(phase_dir, subset)
            if not os.path.isdir(root):
                continue
            if not (os.path.isdir(os.path.join(root, "input")) and os.path.isdir(os.path.join(root, "target"))):
                print(f"[SKIP] Not a pair folder (needs input/ & target/): {root}")
                continue
            print(f"[BUILD] {phase}/{subset}")
            build_split(root, recursive=recursive)


if __name__ == "__main__":
    # Set recursive=True if input/target contain nested subfolders.
    build_realblur_tree(base_dir="datasets/RealBlur", recursive=False)
