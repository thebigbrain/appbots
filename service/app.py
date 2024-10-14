from flask import Flask, jsonify, request

from appbots.datasets.annotation import get_bot_memo
from service.modules import canny, yolo

app = Flask(__name__)


@app.route('/detection/predict', methods=['GET'])
def predict():
    image_id = int(request.args.get('image_id', 0))
    mode = request.args.get('mode', 'canny')
    image = get_bot_memo(image_id)

    boxes = []
    if image is not None:
        if mode == 'canny':
            boxes = canny.predict(image.get('screenshot'))
        elif mode == 'yolo':
            boxes = yolo.predict(image.get('screenshot'))
    return jsonify(dict(image_id=image_id, boxes=boxes))


if __name__ == '__main__':
    app.run(debug=True)
