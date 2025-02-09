from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(data=r"G:\Ahmed\Ai Projects\Ai keyboard\Dataset\DataSplit\data.yaml", epochs=20)


if __name__ == '__main__':
    main()