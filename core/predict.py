from torchvision import models
import json
from . import transform_image
import os
path=os.path.abspath('.')

imagenet_class_index = json.load(open(path+'/core/imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


def get_prediction(image_bytes):
  tensor = transform_image.transform_image(image_bytes=image_bytes)
  outputs = model.forward(tensor)
  _, y_hat = outputs.max(1)
  predicted_idx = str(y_hat.item())
  return imagenet_class_index[predicted_idx]
