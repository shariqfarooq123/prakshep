from django.http import JsonResponse
from django.views import View
from .forms import UploadImageForm
from django.shortcuts import render
import numpy as np
import cv2
from extractor import Extractor

class OcrApi(View):
    """
    Base class for handling image upload
    """
    def post(self,request):
        form = UploadImageForm(request.POST,request.FILES)
        if form.is_valid():
            # handle upload
            image_file = form.cleaned_data['image']
            raw_array = self.file_to_ndarray(image_file)
            img = cv2.imdecode(raw_array,0)

            e = self.get_extractor()
            if e is None:
                data = dict(message="Extractor not implemented error")
                return JsonResponse(data)
            data = e.extract(img)
            if data is None:
                data = dict(message='Failed to extract card, make sure emblem on the card is clearly visible')
                return JsonResponse(data)

            data['message'] = "success"
            return JsonResponse(data)

        else:
            form = UploadImageForm()

        return render(request,'upload.html',{'form':form})

    def get(self,request):
        form = UploadImageForm()
        card = self.get_card_type()
        return render(request,'upload.html',{'form':form,'card':card})

    def file_to_ndarray(self,file):
        return np.asarray(bytearray(file.read()),dtype=np.uint8)


    def get_extractor(self):
        return None

    def get_card_type(self):
        return None

class OcrApiPan(OcrApi):
    # overide extractor
    def get_extractor(self):
        return Extractor(card_type='p')

    def get_card_type(self):
        return "pan"


class OcrApiAadhaar(OcrApi):
    # overide extractor
    def get_extractor(self):
        return Extractor(card_type='a')

    def get_card_type(self):
        return "aadhaar"



def home(request):
    render(request,'home.html')