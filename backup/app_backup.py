'''
Predict image only (done)
'''


from flask import Flask, request, jsonify, render_template, send_file
import io
import cv2
import torch
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

@app.route('/')
def index():
    return render_template('index_backup.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Convert the image to a numpy array
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Make the prediction using the YOLOv5 model
    results = model(img)

    # Process the prediction results
    boxes = results.xyxy[0].tolist()
    class_names = results.names

    # Draw bounding boxes on the image
    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box
        class_name = class_names[int(class_id)]
        label = f"{class_name} {confidence*100:.2f}%"
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (36, 255, 12), 2)
        img = cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Save the annotated image to a byte stream
    _, img_bytes = cv2.imencode('.png', img)
    return send_file(io.BytesIO(img_bytes.tobytes()), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, request, jsonify, render_template, send_file
# import io
# import cv2
# import torch
# import numpy as np
# from PIL import Image

# app = Flask(__name__)

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the image file from the request
#     file = request.files['image']

#     # Convert the image to a numpy array
#     img_bytes = file.read()
#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

#     # Make the prediction using the YOLOv5 model
#     results = model(img)

#     # Process the prediction results
#     boxes = results.xyxy[0].tolist()
#     class_names = results.names

#     # Draw bounding boxes on the image
#     for box in boxes:
#         x1, y1, x2, y2, confidence, class_id = box
#         class_name = class_names[int(class_id)]
#         label = f"{class_name} {confidence*100:.2f}%"
#         img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (36, 255, 12), 2)
#         img = cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

#     # Save the annotated image to a byte stream
#     _, img_bytes = cv2.imencode('.png', img)
#     return send_file(io.BytesIO(img_bytes.tobytes()), mimetype='image/png')

# if __name__ == '__main__':
#     app.run(debug=True)