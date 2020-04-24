import os, sys, cv2
import numpy as np
from tqdm import tqdm
from xml.etree.ElementTree import Element, dump, ElementTree
import time

from modeling.deeplab import *
from dataloaders.utils import decode_segmap
import random
 

def blend(image, board, alpha=.5):

    temp = (board>0).reshape(board.shape)
    beta = 1.0-alpha
    input1 = board
    image[temp] = 0#image[temp]*beta
    input2 = image
    
    res = input2
    res = np.uint8(res)
    return res

def indent(elem, level=0): # https://goo.gl/J8VoDK
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

cls_dict_outside = {1:'container', 2:'slightly rusting', 3:'serious rusting', 4:'slightly deformation',
            5:'serious deformation', 6:'container ID'}
cls_dict_inside = {1:'slightly rusting', 2:'serious rusting', 3:'slightly deformation',
        4:'serious deformation', 6:'oil contamination', 7:'bad sealing', 8:'novel object',
        9:'damaged floor'}

def run(originalImages,pictureType):

    if pictureType == 'outside':
        num_classes = 20+1
        ckptname = 'model_best.pth.tar'
        cls_dict = cls_dict_outside
        # condition=[3, 5]
        condition=[5]
    elif pictureType == 'inside':
        num_classes = 8+1
        ckptname = 'model_best_inside.pth.tar'
        cls_dict = cls_dict_inside
        condition=[2, 7, 8, 9]
    else:
        raise NotImplementedError 


    print('eval mode')
    # device=torch.device("cpu")
    device = torch.device("cuda")
    print('load model...') 
    model = DeepLab(num_classes=num_classes,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=False,
                    freeze_bn=True)
    print('load weights...')
    pth_path=os.path.join('./run', 'pascal', 'deeplab-resnet', ckptname)
    print(pth_path)
    #TODO: Read only state_dict
    # model.load_state_dict(torch.load(pth_path, map_location="cpu")['state_dict'])
    model.load_state_dict(torch.load(pth_path, map_location="cuda:0")['state_dict'])
    model.to(device)
    model.eval()

    # 
    _result = []
    _base_dir = os.getcwd()
    # print(_base_dir)
    # _image_dir = os.path.join(_base_dir, 'JPEGImages')
    # _cat_dir = os.path.join(_base_dir, 'SegmentationClass_Total')
    # _split_dir = os.path.join(_base_dir,'ImageSets', 'Segmentation')

    images = []
    categories = []
    # with open(os.path.join(_split_dir,'val.txt'), "r") as f:
        # lines = f.read().splitlines()

    # _tbar = tqdm(lines)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    ii= 0
    now = int(time.time())*random.randint(0,1000) # random a folder name
    queryPath = os.path.join('images','result',str(now))
    resPathBase = os.path.join(_base_dir,'static',queryPath)
    os.makedirs(resPathBase)
    for originalImage in originalImages:
        # _image = originalImage # os.path.join(_image_dir, line + ".jpg")
        # _cat = os.path.join(_cat_dir, line + ".png")
        # assert os.path.isfile(_image)
        # assert os.path.isfile(_cat)
        
        #image byte convert to open-cv image
        img_np_arr = np.frombuffer(originalImage, np.uint8)
        _img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR) #cv2.imread(_image)
        img = cv2.cvtColor(_img,cv2.COLOR_BGR2RGB) # be used to 
        H, W, _ = img.shape
        w = 513
        stepH = int(H/w)
        stepW = int(W/w)
        kh = -int((H-w)/stepH - w)
        kw = -int((W-w)/stepW - w)
        board = np.zeros((H, W, 3), dtype=np.uint8)
        cls_map = np.zeros((H, W), dtype=np.uint8)
        shows = np.zeros((H, W, 3), dtype=np.uint8)
        a_hn = 0
        a_wn = 0
        for sh in range(0, stepH + 1):
            a_hn = w * sh - kh * sh
            for sw in range(0, stepW + 1):
                a_wn = w * sw - kw * sw
                if a_wn + w >= W:
                    a_wn = W - w
                if a_hn + w >= H:
                    a_hn = H - w
                cropped = img[a_hn:a_hn + w, a_wn:a_wn + w]
                cropped = cropped.astype('float')


                cropped /= 255.0
                cropped -= mean
                cropped /= std
                cropped = np.expand_dims(cropped, axis=0)
                cropped = np.transpose(cropped,(0,3, 1, 2))
                cropped = torch.from_numpy(cropped).float()
                # image = cropped.cpu()
                image = cropped.cuda()

                with torch.no_grad():
                    output = model(image)
                output = torch.max(output[:3], 1)[1].detach().cpu().numpy()
                image = image.cpu().numpy()
                pred = np.squeeze(output)
                #TODO: Show Segmented image, Show accuracy, Show IoU per class
                color_map = decode_segmap(pred,dataset='pascal')
                color_map= color_map*255
                board[a_hn:a_hn + w, a_wn:a_wn + w]=color_map.astype(np.uint8)
                cls_map[a_hn:a_hn + w, a_wn:a_wn + w]=pred.astype(np.uint8)


        #TODO: Resize
        # target = decode_segmap(target, dataset='pascal')
        # target = target*255
        # target = target.astype(np.uint8)
        if W > H:
            h = 500
            w = int(500 * H/W + .5)
        else:
            w = 500
            h = int(500 * H/W + .5)
        _img = cv2.resize(_img, dsize=(w,h))
        board = cv2.resize(board, dsize=(w,h))
        # target = cv2.resize(target, dsize=(w,h))
        defects = np.unique(cls_map)
        if pictureType == 'outisde':
            defects = defects[2:]
            if defects[-1]==6:
                defects = defects[:-1]
        else:
            defects = defects[1:]
        print(defects)

        root = Element("defects")
        filename = Element("filename")
        filename.text =  str(ii) + '.jpg'
        defects_list = Element("list")
        pass_flag = Element("Pass_condition")
        # if len(defects) == 0:
        isPass = False
        if len(defects) == 0:
            pass_flag.text = "pass"
            isPass = True
        else:
            for defect in defects:
                for critical in condition:
                    if defect == critical:
                        pass_flag.text = "non pass"
                        isPass = False
                        break
                    else:
                        pass_flag.text = "pass"
                        isPass = True
                if pass_flag.text == 'non pass':
                    break

        defectsArr = []
        for cls in defects:
            if cls == 0:
                continue
            detect_element = Element("detect")
            detect_element.text = cls_dict[cls]
            defectsArr.append(str(cls))
            defects_list.append(detect_element)

        root.append(filename)
        root.append(pass_flag)
        root.append(defects_list)
        indent(root)
        
        resXmlPath = os.path.join(resPathBase,str(ii)+'.xml')
        ElementTree(root).write(resXmlPath)
        dump(root)
        # print(os.path.join(_base_dir,'inference',line+'.png'))
        # print("{}\n{}".format(_img.shape, board.shape))
        store = np.hstack([_img, board])
        # cv2.imshow("test",store)
        key = cv2.waitKey(0)
        if key == 113:
            sys.exit()
        desPath = os.path.join(resPathBase,str(ii)+'.png')
        cv2.imwrite(desPath,store)
        # image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        _result.append({"image":os.path.join(queryPath,str(ii)+'.png'),"isPass":isPass,"defects":defectsArr,"containerID":''})
        ii=ii+1

    return _result
