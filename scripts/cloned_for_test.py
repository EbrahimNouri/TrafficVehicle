import os
import zipfile
import shutil
import requests
from tqdm import tqdm
from pathlib import Path

# ------------------- تنظیمات -------------------
REPO_ZIP_URL = "https://github.com/pouria-maleki/Iranian-Vehicle-images-dataset-for-detection/raw/main/Iranian_vehicle_dataset.zip"
ZIP_NAME = "Iranian_vehicle_dataset.zip"
EXTRACT_DIR = "temp_extracted"
OUTPUT_DIR = "dataset"
# ------------------------------------------------

def download_file(url, dest):
    """دانلود فایل با نمایش نوار پیشرفت"""
    if os.path.exists(dest):
        print(f"[*] فایل {dest} از قبل وجود دارد. رد می‌شود.")
        return
    print(f"[*] در حال دانلود {url} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    """استخراج زیپ"""
    if os.path.exists(extract_to):
        print(f"[*] پوشه {extract_to} از قبل وجود دارد. پاک می‌شود.")
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    print(f"[*] در حال استخراج {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

def find_images(root):
    """پیدا کردن همه تصاویر با پسوندهای رایج"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                images.append(os.path.join(dirpath, fn))
    return images

def detect_structure(root):
    """
    تشخیص ساختار:
    - اگر زیرپوشه‌هایی با نام car/bus/truck وجود داشت => 'structured'
    - وگرنه => 'flat'
    """
    class_names = {"car", "bus", "truck"}
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() in class_names:
                return "structured"
    return "flat"

def copy_structured(root, output_dir):
    """کپی با حفظ ساختار پوشه‌های کلاس"""
    print("[*] ساختار پوشه‌های کلاس شناسایی شد. در حال کپی ...")
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            if d.lower() in {"car", "bus", "truck"}:
                src_dir = os.path.join(dirpath, d)
                dst_dir = os.path.join(output_dir, d.lower())
                os.makedirs(dst_dir, exist_ok=True)
                for fn in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, fn)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, os.path.join(dst_dir, fn))
    print("[✓] کپی ساختارمند انجام شد.")

def copy_flat(root, output_dir):
    """
    حالت تخت: سعی می‌کنیم از نام فایل یا فایل‌های annotation کلاس را استخراج کنیم.
    اگر annotation پیدا نشد، همه را در پوشه 'unknown' می‌ریزیم تا دستی بررسی کنی.
    """
    print("[*] ساختار تخت شناسایی شد. تلاش برای تفکیک بر اساس نام فایل/annotation ...")
    images = find_images(root)
    print(f"[*] تعداد تصاویر پیدا شده: {len(images)}")

    # ۱) تلاش برای خواندن فایل classes.txt یا annotations.csv
    class_map = {}  # نام فایل -> کلاس
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low == "classes.txt":
                with open(os.path.join(dirpath, fn), "r", encoding="utf-8") as f:
                    classes = [l.strip().lower() for l in f if l.strip()]
                print(f"[*] classes.txt پیدا شد: {classes}")
            if low.endswith(".csv"):
                # فرمت‌های رایج: filename,class  یا  filename,xmin,ymin,xmax,ymax,class
                import csv
                with open(os.path.join(dirpath, fn), "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) < 2:
                            continue
                        fname = row[0].strip()
                        # آخرین ستون را به عنوان کلاس در نظر می‌گیریم
                        cls = row[-1].strip().lower()
                        if cls in {"car", "bus", "truck"}:
                            class_map[fname] = cls
                print(f"[*] از CSV، {len(class_map)} نگاشت استخراج شد.")

    # ۲) کپی بر اساس نگاشت یا نام فایل
    counts = {"car": 0, "bus": 0, "truck": 0, "unknown": 0}
    for img in images:
        fname = os.path.basename(img)
        cls = None

        # اول از class_map
        if fname in class_map:
            cls = class_map[fname]
        else:
            # جستجو در نام فایل
            low = fname.lower()
            for c in ["car", "bus", "truck"]:
                if c in low:
                    cls = c
                    break

        if cls is None:
            cls = "unknown"

        dst_dir = os.path.join(output_dir, cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(img, os.path.join(dst_dir, fname))
        counts[cls] += 1

    print(f"[✓] کپی انجام شد. آمار: {counts}")
    if counts["unknown"] > 0:
        print("[!] برخی تصاویر در پوشه 'unknown' قرار گرفتند. لطفاً دستی بررسی کن.")

def main():
    download_file(REPO_ZIP_URL, ZIP_NAME)
    extract_zip(ZIP_NAME, EXTRACT_DIR)

    structure = detect_structure(EXTRACT_DIR)
    print(f"[*] ساختار تشخیص داده شده: {structure}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if structure == "structured":
        copy_structured(EXTRACT_DIR, OUTPUT_DIR)
    else:
        copy_flat(EXTRACT_DIR, OUTPUT_DIR)

    print(f"[✓] تمام شد. خروجی در پوشه: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()