from flask import Flask, jsonify, request
from core import predict as prd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        result = []
        if 'file' not in request.files:
          return jsonify({'success':False})
        for filename in request.files.getlist('file'):
          image_bytes = filename.read()
          class_id, class_name = prd.get_prediction(image_bytes=image_bytes)
          result.append({'class_id': class_id, 'class_name': class_name})
        return jsonify({"success":True,"data":result})
    else:
      return jsonify({'success':False,"message":'use POST Method please'})    


@app.route('/',methods=['GET'])
def home():
  return jsonify({'hello':'world'})

if __name__ == '__main__':
    app.run()