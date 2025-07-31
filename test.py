from ultralytics import YOLO
import numpy as np
import os

model = YOLO("/mnt/nas4/lsj/Tire-test/runs/train_tire5/weights/best.pt")

# 2. 추론 실행
results = model.predict(
    source='/mnt/nas4/lsj/Tire-test/test-img/test01.jpg',  # 이미지 경로
    imgsz=640,                                    # 이미지 사이즈
    project='/mnt/nas4/lsj/Tire-test/test-predict',            # 결과 저장 프로젝트 폴더
    name='test-result',                          # 결과 저장 폴더 이름
    save=True,                                    # 이미지 저장
    save_txt=True                                 # keypoints 좌표 저장 (원하면)
) 

# 3. 결과 확인 (선택)
for r in results:
    print(f"결과 {r.keypoints[0]}")# 키포인트 정보 출력 (옵션)

result_json = r.to_json()
base_path = "/mnt/nas4/lsj/Tire-test"
save_name = "Tire-result-text"
ext = ".json"
i = 1
while True:
    save_path = os.path.join(base_path, f"{save_name}_{i:02d}{ext}")
    if not os.path.exists(save_path):
        break
    i+=1
with open(save_path, 'w') as f:
    f.write(result_json)

detected_points = r.keypoints[0].xy.cpu().numpy().astype(np.int32)
print(f"검출된 키포인트 좌표: {detected_points}")