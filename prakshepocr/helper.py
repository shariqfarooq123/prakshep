from extractor_manual import extract_info_scene, extract_info_fit
from extractor import Extractor
from scipy.misc import imresize

class ExtractorHandler:
    def __init__(self,mode='fit',card_type='pan',image=None,**kwargs):
        self.mode = mode
        self.card_type = card_type
        if image is None:
            raise ValueError("Image is None")
        self.image = image
        bounds = kwargs.get('bounds')
        if bounds is not None:
            self.bounds = bounds

    def get_data(self):
        data = None
        if self.mode not in ['fit','bounded','raw']:
            return dict(message="ValueError! mode must be one of 'bounds', 'fit' or 'raw'")

        if self.card_type not in ['pan','aadhaar']:
            return dict(message="ValueError! card type must be one of 'pan' or 'aadhar'")

        if self.image is None:
            data = dict(message="ValueError! Image received is NULL")
            return data

        if self.mode == 'fit':
            data = extract_info_fit(self.image,self.card_type[0])


        elif self.mode == 'bounded':
            data = extract_info_scene(self.image,self.bounds,self.card_type[0])

        elif self.mode == 'raw':
            e = Extractor(self.card_type[0])
            data = e.extract(self.image)
            if data is None:
                return dict(message="mode is raw but data is None card type is {}".format(self.card_type[0]))

        if data is not None:
            # data['message'] = "success"
            return data

        else:
            data = dict(message="Something wrong happened! Check your url!")
            return data
