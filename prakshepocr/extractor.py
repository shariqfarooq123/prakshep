import numpy as np
import cv2
from scipy.misc import imresize
from pytesseract import image_to_string
from PIL import Image
from skimage.filters import threshold_local
from skimage.exposure import equalize_adapthist
from skimage.morphology import thin
from skimage.util import invert
from skimage.color import rgb2gray
from settings import BASE_DIR
import re



class Extractor(object):

    def __init__(self,card_type='p',min_match_count=10):
        self.card_type = card_type
        data_dir = BASE_DIR + "/prakshepocr/data/"
        self.MIN_MATCH_COUNT = min_match_count

        if card_type == 'p':
            self.config = dict(name=(0.28, 0.38, None, None), parent=(0.404, 0.507, None, None),
                          dob=(0.51, 0.61, None, None),
                          pan=(0.685, 0.79, None, None))

            pan_template_path = data_dir + "temp_pan.jpg"
            template = cv2.imread(pan_template_path,0)
            self.fields = "name parent dob pan".split()

        elif card_type == 'a':
            self.config = dict(name=(0.346, 0.434, 0.345, None), dob=(0.441, 0.508, 0.345, None),
                          gender=(0.518, 0.613, 0.345, 0.686), uid=(0.726, 0.824, 0.345, 0.713))

            aadhaar_template_path = data_dir + "temp_aadh.jpg"
            template = cv2.imread(aadhaar_template_path,0)
            self.fields = "name dob uid gender".split()
        else:
            raise ValueError("Cannot identify card type ({})".format(card_type))

        if template is None:
            raise IOError("Couldn't read template")

        # template = equalize_adapthist(template)*255
        # template = np.uint8(template)
        self.template = template


    def _skeleton(self, img):
        image = invert(img)
        skl = thin(image / 255, max_iter=1).astype('int')
        skull = 1 - skl
        return np.uint8(skull * 255)


    def _get_roi_map(self,img, pad=0):
        h, w = img.shape
        if self.card_type == 'p':
            img = img[:, pad:w / 2]

        # print "fields:\n", self.fields
        ceiling = lambda x, i: int(np.ceil(x * i)) if x is not None else None
        roi_map = {field: img[ceiling(self.config[field][0], h): ceiling(self.config[field][1], h), \
                          ceiling(self.config[field][2], w): ceiling(self.config[field][3], w)] for field in self.fields}
        # print "roi\n", roi_map
        return roi_map


    def str_img(self,img):
        img = self._skeleton(img)
        pimg = Image.fromarray(img)
        text = image_to_string(pimg)
        return text


    def _parse_card(self,img):
        block_size = 35
        adaptive_thresh = threshold_local(img, block_size, offset=10)
        binary_adaptive = (img > adaptive_thresh).astype('int')
        im = np.uint8(binary_adaptive * 255)
        roi_map = self._get_roi_map(im)
        parsed_data = {field: self.str_img(img_field) for field, img_field in roi_map.items()}
        return parsed_data

    def _extract_card(self,img,template_size,scene_size):

        template = imresize(self.template, template_size) # template (emblem)
        img = imresize(img, scene_size) # scene
        # img = equalize_adapthist(img)*255
        # img = np.uint8(img)



        # Initiate SIFT detector
        sift = cv2.SIFT()

       # print "Computing sift..."
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(img, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        #print "FLANN matching..."
        matches = flann.knnMatch(des1, des2, k=2)
        #print "done"
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, hmask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            warped = cv2.warpPerspective(img, H, (template.shape[1], template.shape[0]))
            return warped
        else:
         #   print "Not enough matches are found - %d/%d" % (len(good), self.MIN_MATCH_COUNT)
            return None

    def extract(self,img):
        if type(img) == str:
            img = cv2.imread(img,0)
        elif type(img) == np.ndarray:
            if len(img.shape) == 3 and img.shape[-1] == 3:
                # img is rgb color image, convert to grayscale
                img = rgb2gray(img)
        else:
            raise ValueError("Only filename or ndarray is supported for argument 'img'")


        # Try various aspect ratios
        # SIFT is scale invariant, not aspect ratio invariant!!
        template_sizes = [(512,1024),(1024,1024),(1024,512),(256,1024),(1024,256),(885,1024)]
        scene_size = (1024,1024)

        for template_size in template_sizes:
          #  print "Trying template_size = {} and scene_size = {}".format(template_size,scene_size)
            card = self._extract_card(img,template_size,scene_size)
            if card is not None:
                break
        else:
            # This is a for-else loop (if you're not familiar)
            # (Control is brought here when) Above for loop completed normally without encountering break statement
            # Couldn't extract card!!
           # print "Failed to extract card! Make sure emblem is clearly visible!"
            return None


        data = self._parse_card(card)

        data = self.filter(data)
        return data

    def _filter_field(self,pattern,field,**kwargs):
       # print "original field:\n",field
        matches = re.findall(pattern,field)
        length = len(matches)

        minlen = kwargs.get("minlen")
        if minlen is not None:
            length = minlen

        if matches is not None and len(matches) == length:
                field = " ".join(matches)

       # print "\n\nreturn field:\n",field
        if field == "":
            field = "None, Retry to get the field properly!"
        return field



    def filter(self,data):
        # ignore unicode
        data = {x:y.encode('ascii','ignore') for x,y in data.items()}

        patterns = dict(gender="MALE|FEMALE",
                        uid="[0-9]{4}",
                        parent="[A-Za-z]+",
                        name="[A-Za-z]+",
                        pan="[A-Z0-9]",
                        dob="[0-9/]+")

        getlen = lambda x: 3 if x == 'uid' else None
        data = {x:self._filter_field(patterns[x],y,minlen=getlen(x)) for x,y in data.items()}
        if 'pan' in data.keys():
            data['pan'] = "".join(data['pan'].split())
        return data
