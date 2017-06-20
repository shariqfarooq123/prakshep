import glob
import numpy as np
import cv2
from scipy.misc import imresize
from pytesseract import image_to_string
from PIL import Image
from skimage.filters import threshold_local
from skimage.transform import pyramid_expand, pyramid_reduce
from skimage.morphology import thin
from skimage.util import invert
from skimage.color import rgb2gray
from settings import BASE_DIR
import re
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist, adjust_gamma, equalize_adapthist, adjust_sigmoid
from skimage.restoration import denoise_bilateral as denoise


class Extractor(object):
    """
    Helper class for card extraction
    """

    def __init__(self, card_type='p', min_match_count=10, verbose=0, interactive=False):
        self.verbose = verbose
        self.interactive = interactive
        self.card_type = card_type
        self.data_dir = BASE_DIR + "/prakshepocr/data/"
        self.MIN_MATCH_COUNT = min_match_count

        if card_type == 'p':
            # get the geometrical parameters for pan card
            # each param stores pixel bounds in the form (row_start, row_end, col_start, col_end)
            self.config = dict(name=(0.28, 0.38, None, None), parent=(0.404, 0.507, None, None),
                               dob=(0.51, 0.61, None, None),
                               pan=(0.685, 0.79, None, None))

            pan_template_path = self.data_dir + "temp_pan.jpg"
            template = cv2.imread(pan_template_path, 0)
            self.fields = "name parent dob pan".split()

        elif card_type == 'a':
            # get the geometrical parameters for aadhaar card
            # each param stores pixel bounds in the form (row_start, row_end, col_start, col_end)
            self.config = dict(name=(0.346, 0.434, 0.345, None), dob=(0.441, 0.508, 0.345, None),
                               gender=(0.518, 0.613, 0.345, 0.686), uid=(0.726, 0.824, 0.345, 0.713))

            aadhaar_template_path = self.data_dir + "temp_aadh.jpg"
            template = cv2.imread(aadhaar_template_path, 0)
            self.fields = "name dob uid gender".split()
        else:
            raise ValueError("Cannot identify card type ({})".format(card_type))

        if template is None:
            raise IOError("Couldn't read template")

        self.template = template

    def _skeleton(self, img):
        """
        Obtain the skeleton image of a given image
        :param img: ndarray - input image 
        :return: skeleton binary image
        """
        image = invert(img)
        skl = thin(image / 255, max_iter=1).astype('int')
        skull = 1 - skl
        return np.uint8(skull * 255)

    def _get_roi_map(self, img, pad=0):
        """
        Get a mapping in the form of a python dict where
        key - string; field name
        value - ndarray; cropped image which contains only the specified field
        :param img: input image
        :param pad: number of columns to skip
        :return: dict; mapping from field to cropped section of the card
        """
        h, w = img.shape
        if self.card_type == 'p':
            img = img[:, pad:w / 2]

        # print "fields:\n", self.fields
        ceiling = lambda x, i: int(np.ceil(x * i)) if x is not None else None
        roi_map = {field: img[ceiling(self.config[field][0], h): ceiling(self.config[field][1], h), \
                          ceiling(self.config[field][2], w): ceiling(self.config[field][3], w)] for field in
                   self.fields}
        # print "roi\n", roi_map
        return roi_map

    def str_img(self, img):
        """
        Convert image to string
        :param img: ndarray; input image
        :return: string; text found in the image
        """
        img = self._skeleton(img)
        pimg = Image.fromarray(img)
        text = image_to_string(pimg)
        return text

    def _parse_card(self, img):
        """
        parse the different fields out of given fitted card image
        :param img: ndarray; fitted card image
        :return: python dict; mapping field names to field values
        """
        img = imresize(img,(512,1024))
        block_size = 35
        adaptive_thresh = threshold_local(img, block_size, offset=10)
        binary_adaptive = (img > adaptive_thresh).astype('int')
        im = np.uint8(binary_adaptive * 255)
        roi_map = self._get_roi_map(im)
        parsed_data = {field: self.str_img(img_field) for field, img_field in roi_map.items()}
        return parsed_data

    def threshold_loc(self, img):
        """
        obtain the adaptive threshold of an image
        :param img: 
        :return: 
        """
        t = threshold_local(img, 35, offset=10)
        bin = img > t
        bin = bin.astype('int')
        bin = np.uint8(bin * 255)
        return bin

    def _extract_card(self, img, template_size, scene_size):
        """
        Extract the card from a given image
        :param img: ndarray; input scene
        :param template_size: size to which template is to be resized 
        :param scene_size: size to which scene is to be resized
        :return: fitted card image
        """
        template = imresize(self.template, template_size)  # template (emblem)
        img = imresize(img, scene_size)  # scene

        # transformation = lambda x: np.uint8(denoise(x,multichannel=False)*255)
        # img, template = map(transformation,[img,template])

        # Initiate SIFT detector
        # sift = cv2.ORB() # results not good
        sift = cv2.SIFT()

        self.log("Computing sift...")
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(img, None)

        FLANN_INDEX_KDTREE = 1
        # FLANN_INDEX_LSH = 6 # used this with ORB


        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # index_params = dict(algorithm=FLANN_INDEX_LSH,table_number=12,key_size=20,multi_probe_level=2) # ORB
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.log("FLANN matching...")
        matches = flann.knnMatch(des1, des2, k=2)
        self.log("done")

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if self.interactive:
            self.draw_matches(img, kp2, template, kp1, good)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, hmask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            warped = cv2.warpPerspective(img, H, (template.shape[1], template.shape[0]))
            return warped
        else:
            self.log("Not enough matches are found - %d/%d" % (len(good), self.MIN_MATCH_COUNT))
            return None

    def lower_quality_gen(self, img, max_count=3):
        up = img
        for i in xrange(max_count + 1):
            yield up
            reduced = pyramid_reduce(up)
            up = pyramid_expand(reduced)

    def draw_matches(self, img1, kp1, img2, kp2, matches, color=None):
        """
        Taken from git
        drawMatches function is present in opencv 3.0+
        Draws lines between matching keypoints of two images.  
        Keypoints not in a matching pair are not drawn.
        Places the images side by side in a new image and draws circles 
        around each keypoint, with line segments connecting matching pairs.
        You can tweak the r, thickness, and figsize values as needed.
        Args:
            img1: An openCV image ndarray in a grayscale or color format.
            kp1: A list of cv2.KeyPoint objects for img1.
            img2: An openCV image ndarray of the same format and with the same 
            element type as img1.
            kp2: A list of cv2.KeyPoint objects for img2.
            matches: A list of DMatch objects whose trainIdx attribute refers to 
            img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
            color: The color of the circles and connecting lines drawn on the images.  
            A 3-tuple for color images, a scalar for grayscale images.  If None, these
            values are randomly generated.  
        """
        # We're drawing them side by side.  Get dimensions accordingly.
        # Handle both color and grayscale images.
        if len(img1.shape) == 3:
            new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
        new_img = np.zeros(new_shape, type(img1.flat[0]))
        # Place images onto the new image.
        new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
        new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

        # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
        r = 15
        thickness = 2
        if color:
            c = color

        matches = np.ravel(matches)
        for m in matches:

            # Generate random color for RGB/BGR and grayscale images as needed.
            if not color:
                c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
            # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
            # wants locs as a tuple of ints.
            end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
            end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
            cv2.line(new_img, end1, end2, c, thickness)
            cv2.circle(new_img, end1, r, c, thickness)
            cv2.circle(new_img, end2, r, c, thickness)

        plt.figure(figsize=(15, 15))
        plt.imshow(new_img, 'gray')
        plt.show()

    def extract(self, img):
        if type(img) == str:
            img = cv2.imread(img, 0)


        elif type(img) == np.ndarray:
            if len(img.shape) == 3 and img.shape[-1] == 3:
                # img is rgb color image, convert to grayscale
                img = rgb2gray(img)
        else:
            raise ValueError("Only filename or ndarray is supported for argument 'img'")

        # Try various aspect ratios
        # SIFT is scale invariant, not aspect ratio invariant!!
        template_sizes = [(512, 1024), (1024, 1024), (1024, 512), (256, 1024), (1024, 256), (885, 1024)]
        scene_size = (1024, 1024)

        # Bring down quality step by step
        # images = self.lower_quality_gen(img)
        # for image in images:


        for template_size in template_sizes:
            self.log("Trying template_size = {} and scene_size = {}".format(template_size, scene_size))
            card = self._extract_card(img, template_size, scene_size)
            if card is not None:
                break

        # else:
        #     continue
        #     break

        else:
            # This is a for-else loop (if you're not familiar)
            # (Control is brought here when) Above for loop completed normally without encountering break statement
            # Couldn't extract card!!
            self.log("Failed to extract card! Make sure emblem is clearly visible!")
            return dict(message="failed to extract card")
            # return None

        data = self._parse_card(card)
        status = "None" if data is None else "Present"
        filtered = self.filter(data)
        return dict(message="succes {}".format(status),raw=data,filtered=filtered)

    def _filter_field(self, pattern, field, **kwargs):
        """
        filter the extracted fields using regular expression
        :param pattern: 
        :param field: 
        :param kwargs: 
        :return: 
        """
        # print "original field:\n",field
        matches = re.findall(pattern, field)
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

    def filter(self, data):
        """
        filter the data obtained after extraction
        :param data: 
        :return: dict; filtered data
        """

        # ignore unicode
        data = {x: y.encode('ascii', 'ignore') for x, y in data.items()}

        patterns = dict(gender="MALE|FEMALE",
                        uid="[0-9]{4}",
                        parent="[A-Za-z]+",
                        name="[A-Za-z]+",
                        pan="[A-Z0-9a-z]",
                        dob="[0-9/]+")

        getlen = lambda x: 3 if x == 'uid' else None
        data = {x: self._filter_field(patterns[x], y, minlen=getlen(x)) for x, y in data.items()}
        if 'pan' in data.keys():
            data['pan'] = "".join(data['pan'].split())
        return data

    def log(self, string):
        if self.verbose == 1:
            print string


def main():
    Ext = Extractor(card_type='p', verbose=1, interactive=True)
    data = Ext.extract(Ext.data_dir +'test2.jpg')
    print data


if __name__ == "__main__":
    main()
