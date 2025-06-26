import os
from PIL import Image


def compress_image(input_path, output_path, quality=70):
    try:
        with Image.open(input_path) as img:
            # 转换为 RGB 以避免 PNG 透明通道导致的问题
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(output_path, optimize=True, quality=quality)
            print(f"压缩成功: {output_path}")
    except Exception as e:
        print(f"跳过无法处理的文件: {input_path}，原因：{e}")


def compress_folder_images(input_folder, output_folder, quality=70):
    image_extensions = ('.jpg', '.jpeg', '.png')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file)
                compress_image(input_path, output_path, quality)


# 示例用法
if __name__ == "__main__":
    input_folder = r"C:\Users\WH_Ha\Desktop\picture"
    output_folder = r"C:\Users\WH_Ha\Desktop\pic2"
    quality = 70  # 质量范围是 1（最差）到 95（最好），建议设置在 60~80

    compress_folder_images(input_folder, output_folder, quality)
