import json
import os

JSON_PATH = 'Dataset/uitvic_dataset/uitvic_captions_train2017.json'


def find_caption(image_filename):
    print(f"--> Đang tìm caption cho ảnh: {image_filename}")

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        image_id = int(os.path.splitext(image_filename)[0].split('_')[-1])
    except:
        print("Lỗi: Tên file không đúng định dạng chuẩn COCO (ví dụ: 000000123456.jpg)")
        return

    found = False
    print("\n=== CÁC CÂU MODEL ĐÃ ĐƯỢC HỌC VỀ ẢNH NÀY ===")
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            print(f" {ann['caption']}")
            found = True

    if not found:
        print(" Không tìm thấy caption nào cho ID này.")


if __name__ == "__main__":
    img_name = input("Nhập tên file ảnh (VD: 000000123456.jpg): ")
    find_caption(img_name)