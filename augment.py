import os
import cv2
import time
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Rotate, RandomScale

# 경로 설정
image_dir = "data_add/validation"  # 원본 이미지 경로
output_image_dir = "data_add/validation"  # 증강된 이미지 저장 경로

# 폴더 생성
os.makedirs(output_image_dir, exist_ok=True)

# 증강 횟수 설정
augmentation_count = 30

# 증강 파이프라인 설정
transform = Compose([
    Rotate(limit=60, p=0.3),          # 회전
    RandomScale(scale_limit=0.2, p=0.5)  # 크기 조정
])

# 이미지 증강 수행
for img_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, img_file)
    if not os.path.isfile(image_path):
        continue

    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 {img_file} 로드 실패")
        continue

    for i in range(augmentation_count):
        try:
            augmented = transform(image=image)['image']

            # 증강된 이미지 저장
            output_image_name = f"{os.path.splitext(img_file)[0]}_aug_{i+1}.jpg"
            output_image_path = os.path.join(output_image_dir, output_image_name)
            cv2.imwrite(output_image_path, augmented)
            print(f"이미지 저장 완료: {output_image_name}")

        except Exception as e:
            print(f"증강 중 오류 발생: {e}")

print("이미지 증강 완료!")
