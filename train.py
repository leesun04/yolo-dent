from ultralytics import YOLO

#model = YOLO("/mnt/nas4/lsj/Tire-test/runs/train_tire10/weights/best.pt")
model = YOLO("/mnt/nas4/lsj/yolo-dent/models/kpt-model.yaml")
print("모델 정보")
model.info()


model.train(
    data="/mnt/nas4/lsj/yolo-dent/data/data.yaml",
    epochs = 200,
    imgsz = 640,
    batch = 5,
    project = "/mnt/nas4/lsj/yolo-dent/runs",
    name="train_tire"
)