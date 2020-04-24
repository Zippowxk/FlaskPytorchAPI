import numpy as np
import cv2
import os, sys
from tqdm import tqdm

def ind2str(i):
    if i<10:
        return '0'+str(i)
    else:
        return str(i)

def formateImages(originImages):
    _currentPath = os.path.split(os.path.realpath(__file__))
    print(_currentPath)

    _dir = os.path.join(_currentPath,'material/originals')
    _dst = os.path.join(_currentPath,'material/resized')
    # _dst = '/home/minwoo/datasets/VOCdevkit/Container/resized'
    _img = os.path.join(_dir, "JPEGImages")
    _seg = os.path.join(_dir, "rawPNG")
    _res_img = os.path.join(_dst, "JPEGImages_resized")
    _dst_img = os.path.join(_dir, "JPEGImages_resized")
    _dst_seg = os.path.join(_dir, "SegmentationClass")
    print(_img)
    print(_seg)

    list_img = os.listdir(_img)
    list_seg = list(sorted(os.listdir(_seg)))
    flg=False
    flg2=True
    if flg:
        print('----Image Resize Processing...----')
        _titer = tqdm(list_img)
        for ii, fname in enumerate(_titer):
            if os.path.isfile(os.path.join(_res_img,fname)):
                continue
            img = cv2.imread(os.path.join(_img,fname))
            if img is not None:
                H, W, _ = img.shape
                if H > W:
                    h = 500
                    w = int(500 * W/H+.5)
                else:
                    w = 500
                    h = int(500 * H/W+.5)
                img = cv2.resize(img, dsize=(w,h))

                cv2.imwrite(os.path.join(_dst_img,fname), img)
            else:
                print('file not opened: {}'.format(fname))

    if flg2:
        print('----Segmentation Classes Processing...---')
        titer = tqdm(list_seg)
        for i, fname in enumerate(titer):
            out_file = os.path.join(_dst_seg,fname)
            if os.path.isfile(out_file):
                continue
            img = cv2.imread(os.path.join(_seg,fname),cv2.IMREAD_UNCHANGED)
            if img is not None :
                if img.shape[2] is not 4:
                    print('image color:{}'.format(img.shape))
                    print(os.path.join(_seg,fname))
                    sys.exit()
                H, W, _ = img.shape
                r=np.zeros((H,W))
                g=np.zeros((H,W))
                b=np.zeros((H,W))
                
                for c in range(0,3):
                    r[(img[:,:,c]==0)*(img[:,:,3]>0)]=255
                    g[(img[:,:,c]==0)*(img[:,:,3]>0)]=255
                    b[(img[:,:,c]==0)*(img[:,:,3]>0)]=255
                for c in range(0,3):
                    r[(img[:,:,c]>0)*(img[:,:,3]>0)]=255
                    g[(img[:,:,c]>0)*(img[:,:,3]>0)]=255
                    b[(img[:,:,c]>0)*(img[:,:,3]>0)]=255
                img[(img[:,:,3]==0)]=0

                rgb = np.zeros((H,W,3))
                rgb[:,:,0] = r
                rgb[:,:,1] = g
                rgb[:,:,2] = b
                cv2.imwrite(os.path.join(_dst_seg,fname), rgb)
            else:
                print('file not opened: {}'.format(fname))
    else:
        print('false')
