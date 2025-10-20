# prepare_ctc_unet_2d_fixed.py
from pathlib import Path
import re, shutil
from PIL import Image
import sys


ROOT = Path("Fluo-N2DH-GOWT1")     # dataset root shown in your screenshot
OUT_IMG = Path("data/images")
OUT_MSK = Path("data/masks")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MSK.mkdir(parents=True, exist_ok=True)

BINARY_MASKS = True   # True: (mask>0)->1 ; False: keep instance IDs (uint16)
COPY_IMAGES   = True

rx_t   = re.compile(r"^t(\d{3,4})\.tif$", re.IGNORECASE)
rx_seg = re.compile(r"^man_seg(\d{3,4})\.tif$", re.IGNORECASE)

def collect_raw_frames(seq_dir: Path):
    frames = {}
    for p in sorted(seq_dir.glob("t*.tif")):
        m = rx_t.match(p.name)
        if m:
            frames[int(m.group(1))] = p
    return frames

def collect_seg(seg_dir: Path):
    frames = {}
    if not seg_dir.is_dir():
        return frames
    for p in sorted(seg_dir.glob("man_seg*.tif")):
        m = rx_seg.match(p.name)
        if m:
            frames[int(m.group(1))] = p
    return frames

def write_mask(src: Path, dst: Path, binary=True):
    # Use opencv-python instead of PIL to avoid crash
    try:
        import cv2
        arr = cv2.imread(str(src), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Cannot read {src}")

        if binary:
            # Convert to binary: any pixel > 0 becomes 255
            arr = ((arr > 0) * 255).astype('uint8')

        cv2.imwrite(str(dst), arr)
    except ImportError:
        # Fallback: just copy the file if opencv not available
        print(f"    WARNING: opencv not installed, copying mask as-is")
        shutil.copy2(src, dst)

def main():
    # sequences are numeric dirs at ROOT level: 01, 02, ...
    seq_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and d.name.isdigit()],
                      key=lambda p: int(p.name))
    if not seq_dirs:
        print(f"Aucune séquence numérique trouvée sous {ROOT.resolve()}")
        return

    n_pairs = 0
    for seq_dir in seq_dirs:
        seq_id = seq_dir.name             # e.g. "01"
        # RAW frames: ROOT/01/t000.tif ...
        raw = collect_raw_frames(seq_dir)

        # GT/SEG and ST/SEG are at ROOT level: ROOT/01_GT/SEG, ROOT/01_ST/SEG
        gt_seg_dir = ROOT / f"{seq_id}_GT" / "SEG"
        st_seg_dir = ROOT / f"{seq_id}_ST" / "SEG"
        gt_seg = collect_seg(gt_seg_dir)
        st_seg = collect_seg(st_seg_dir)

        print(f"[{seq_id}] raw:{len(raw)} | GT:{len(gt_seg)} | ST:{len(st_seg)}")

        # prefer GT, fallback to ST when GT missing
        masks = dict(gt_seg)
        for k, v in st_seg.items():
            masks.setdefault(k, v)

        common = sorted(set(raw.keys()) & set(masks.keys()))
        if not common:
            print(f"[{seq_id}] aucune paire image/mask trouvée (GT>ST).")
            continue

        print(f"[{seq_id}] {len(common)} paires appariées (GT>ST). Exemple: T={common[0]:03d}")
        sys.stdout.flush()  # Force output before potential crash

        for T in common:
            print(f"  Traitement T={T}...")
            sys.stdout.flush()
            img_src = raw[T]
            msk_src = masks[T]

            # Format: seq_tT.tif (e.g., 01_t000.tif)
            T_digits = len(str(T).zfill(3))  # 3 or 4 digits depending on dataset
            base = f"{seq_id}_t{T:0{T_digits}d}.tif"
            img_dst = OUT_IMG / base
            msk_dst = OUT_MSK / base

            try:
                print(f"    Copie image: {img_src} -> {img_dst}")
                sys.stdout.flush()

                if COPY_IMAGES:
                    # Remove destination if it exists to avoid permission errors
                    if img_dst.exists():
                        img_dst.chmod(0o666)  # Ensure writable
                        img_dst.unlink()
                    shutil.copy2(img_src, img_dst)

                print(f"    Traitement masque: {msk_src} -> {msk_dst}")
                sys.stdout.flush()

                # Remove destination mask if it exists
                if msk_dst.exists():
                    msk_dst.chmod(0o666)
                    msk_dst.unlink()

                print(f"    Ouverture du masque avec PIL...")
                sys.stdout.flush()
                write_mask(msk_src, msk_dst, binary=BINARY_MASKS)

                n_pairs += 1
                if n_pairs % 10 == 0:
                    print(f"  ... {n_pairs} paires traitées")
            except Exception as e:
                print(f"ERREUR sur {base}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"OK: {n_pairs} paires écrites dans data/images & data/masks "
          f"({ 'binaires' if BINARY_MASKS else 'instances' }).")

if __name__ == "__main__":
    main()
