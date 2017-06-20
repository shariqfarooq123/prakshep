from extractor import Extractor
import cv2
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

def extract_info_scene(scene_img,bounds,card_type='p'):
    src_pts = np.float32([[0,0],[1024,0],[1024,512],[0,512]])
    bounds = np.float32(bounds)
    M = cv2.getPerspectiveTransform(bounds,src_pts)
    warped = cv2.warpPerspective(scene_img,M,(1024,512))
    data = extract_info_fit(warped,card_type=card_type)
    return data

def extract_info_fit(card_img, card_type='p'):
    card_img = imresize(card_img,(512,1024))
    ext = Extractor(card_type=card_type)
    raw = ext._parse_card(card_img)
    # raw = {x: y.encode('ascii', 'ignore') for x, y in data.items()}
    filtered = ext.filter(raw)
    data = dict(raw=raw,filtered=filtered)
    return data


def main():
    im = cv2.imread("data/chirag.jpg",0)
    bounds = np.float32([[117,78],[863,69],[892,539],[104,561]])
    data = extract_info_scene(im,bounds,'p')
    print data


if __name__ == "__main__":
    main()