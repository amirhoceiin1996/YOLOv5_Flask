from flask import Flask, request, render_template, Response
import cv2
import torch
import numpy as np

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['image']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    boxes = results.xyxy[0].tolist()
    class_names = results.names

    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box
        class_name = class_names[int(class_id)]
        label = f"{class_name} {confidence*100:.2f}%"
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (36, 255, 12), 2)
        img = cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    _, img_bytes = cv2.imencode('.png', img)
    return Response(img_bytes.tobytes(), mimetype='image/png')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            boxes = results.xyxy[0].tolist()
            class_names = results.names
            for box in boxes:
                x1, y1, x2, y2, confidence, class_id = box
                class_name = class_names[int(class_id)]
                label = f"{class_name} {confidence*100:.2f}%"
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (36, 255, 12), 2)
                frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/predict_webcam')
def predict_webcam():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)
