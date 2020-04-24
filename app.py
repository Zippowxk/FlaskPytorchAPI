from flask import Flask, jsonify, request, render_template,redirect,url_for
from core import predict as prd
import eval
from flask_apscheduler import APScheduler
import shutil
import os
class Config(object):  # 创建配置，用类
    # 任务列表
    JOBS = [  
        {  # 清理任务
            'id': 'job',
            'func': 'app:method_clear', # 方法名
            'trigger': 'cron', # interval表示循环任务
            'hour': 2,
            # 'minute':42,
            # 'second':30,
        }
    ]
# 定点清除文件 
def method_clear():
    print('delete files')
    shutil.rmtree('./static/images/result')


app = Flask(__name__, static_url_path='')
app.config.from_object(Config())
scheduler=APScheduler()
scheduler.init_app(app)
scheduler.start() 

@app.route('/')
def index():
    return redirect('/index.html')


# redirect to index.html when file not found 
_notfound=0
@app.route('/404')
def notFoundFile():
    global _notfound  
    _notfound=1
    return redirect('/index.html')

# redirect to index.html when dir not found 
@app.route('/405')
def notFoundDir():
    global _notfound  
    _notfound=0
    return redirect('/index.html')

@app.route('/clean', methods=['GET'])
def clearFiles():

    return jsonify({'success':True,"message":'start clean'})    


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


@app.route('/predictContainer', methods=['POST'])
def predictContainer():
    global _notfound
    if request.method == 'POST' and _notfound==0:
        os.system('sh dd.sh')
        if 'file' not in request.files:
          return jsonify({'success':False})
        images = []
        images_inside = []
        for filename in request.files.getlist('file'):
          image_bytes = filename.read()
          images.append(image_bytes)
        for filename in request.files.getlist('file_inside'):
          image_bytes_in = filename.read()
          images_inside.append(image_bytes_in)

        print('here we go')
        result_outside = eval.run(originalImages=images,pictureType='outside')
        result_inside = eval.run(originalImages=images_inside,pictureType='inside')
        return jsonify({"success":True,"data":{'outside':result_outside,'inside':result_inside}})
    else:
      return jsonify({'success':False,"message":'use POST Method please'}) 

if __name__ == '__main__':
    app.run()