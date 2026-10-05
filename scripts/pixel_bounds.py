import os
from PIL import Image


def find_smallest_dimensions(root_dir):
    min_width = None
    min_height = None
    min_width_file = None
    min_height_file = None

    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')
    count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(extensions):
                continue

            filepath = os.path.join(dirpath, filename)

            try:
                with Image.open(filepath) as img:
                    width, height = img.size  # (عرض, ارتفاع)
            except:
                continue

            count += 1

            if min_width is None or width < min_width:
                min_width = width
                min_width_file = filepath

            if min_height is None or height < min_height:
                min_height = height
                min_height_file = filepath

    return {
        'min_width': min_width,
        'min_width_file': min_width_file,
        'min_height': min_height,
        'min_height_file': min_height_file,
        'count': count,
    }


if __name__ == "__main__":
    dataset_path = r"C:\Users\viroo\PycharmProjects\TrafficVehicle\dataset"

    result = find_smallest_dimensions(dataset_path)

    print("=" * 60)
    print(f"📊 تعداد عکس‌ها: {result['count']}")
    print("=" * 60)
    print(f"کوچکترین عرض (width)  = {result['min_width']} px")
    print(f"   → {result['min_width_file']}")
    print()
    print(f"کوچکترین ارتفاع (height) = {result['min_height']} px")
    print(f"   → {result['min_height_file']}")
    print("=" * 60)