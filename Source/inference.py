import torch.nn.functional as F
import torch
import cv2
import os
import glob
import config
from torchvision import models, transforms
from model import CLIP, text_encoder_base, tokenizer
import matplotlib.pyplot as plt


MODEL_PATH = 'weights/clip_vietnamese_ep10.pth'
IMAGE_FOLDER = 'Dataset/uitvic_dataset/coco_uitvic_train/coco_uitvic_train'


def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path):
    print(f"--> Đang khởi động và load model...")
    resnet = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()
    model = CLIP(vision_encoder=resnet, text_encoder=text_encoder_base)

    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)
    model = model.to(config.DEVICE)
    model.eval()
    return model


def search_image_interactive(model, img_folder):

    print("--> Đang quét kho ảnh... (Vui lòng đợi xíu)")
    image_paths = glob.glob(os.path.join(img_folder, "*.jpg"))[:500]

    if not image_paths:
        print("LỖI: Không tìm thấy ảnh nào trong folder!")
        return

    print(f"--> Đã nạp xong {len(image_paths)} ảnh vào bộ nhớ.")


    while True:
        print("\n" + "=" * 40)
        query_text = input("👉 Nhập mô tả ảnh bạn muốn tìm (hoặc gõ 'exit' để thoát): ")

        if query_text.lower() in ['exit', 'quit', 'thoat']:
            print("Tạm biệt!")
            break

        if not query_text.strip():
            continue

        print(f"--> Đang tìm: '{query_text}'...")


        inputs = tokenizer([query_text], padding=True, truncation=True, max_length=config.MAX_LENGTH,
                           return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            text_outputs = model.text_encoder(**inputs)
            text_features = text_outputs.last_hidden_state[:, 0, :]
            text_embeddings = model.text_project(text_features)


            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)


        scores = []
        transform = get_transforms()

        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img).unsqueeze(0).to(config.DEVICE)

                with torch.no_grad():
                    image_features = model.vision_encoder(img_tensor)
                    image_embeddings = model.vis_project(image_features)


                    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

                    score = (image_embeddings @ text_embeddings.T).item()
                    scores.append((img_path, score))
            except:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            print("Không tìm được ảnh nào phù hợp.")
            continue

        print(f"\n--- KẾT QUẢ CHO: '{query_text}' ---")
        for i in range(min(5, len(scores))):
            path, score = scores[i]
            filename = os.path.basename(path)
            print(f"Hạng {i + 1}: {filename} - Score: {score:.5f}")

        best_img_path, best_score = scores[0]

        print(f"--> TÌM THẤY! (Độ khớp: {best_score:.4f})")
        print(f"File: {os.path.basename(best_img_path)}")


        img = cv2.imread(best_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Tìm: '{query_text}'\nĐộ khớp: {best_score:.4f}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    search_image_interactive(model, 'Dataset/uitvic_dataset/coco_uitvic_train/coco_uitvic_train')