1. 使用python 3版本的venv开启虚拟环境
```
  python3 -m venv venv
  source venv/bin/activate
``` 

2. 安装依赖库
```pip3 install torch torchvision Flask==1.0.3```
或者使用
```
pip3 install requirement.txt
```

3. 启动Web服务
``` FLASK_ENV=development FLASK_APP=app.py flask run```
