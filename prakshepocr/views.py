from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views import View
from .forms import UploadImageForm,BrowsableForm
from django.shortcuts import render
import numpy as np
import cv2
from helper import ExtractorHandler
import json


def file_to_ndarray(file):
    return np.asarray(bytearray(file.read()),dtype=np.uint8)

def file_to_image(file):
    raw_array = file_to_ndarray(file)
    img = cv2.imdecode(raw_array, 0)
    return img

class OcrApi(View):
    """
    Base class for handling image upload in pure JSON
    """
    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super(OcrApi, self).dispatch(request, *args, **kwargs)

    def post(self,request,**kwargs):
        form = UploadImageForm(request.POST,request.FILES)
        if form.is_valid():
            # handle upload
            image_file = form.cleaned_data['image']
            raw_array = file_to_ndarray(image_file)
            img = cv2.imdecode(raw_array,0)
            bounds = kwargs.get('bounds')
            if bounds is not None:
                try:
                    bounds = np.float32(json.loads(bounds))
                except ValueError as e:
                    data = dict(message=e.message+"\nCheck your bounds!")
                    JsonResponse(data)

            eh = ExtractorHandler(kwargs.get('mode'),kwargs.get('card_type'),img,bounds=bounds)
            data = eh.get_data()
            return JsonResponse(data)

        else:
            return JsonResponse(dict(message="Invalid form data"))
            # form = UploadImageForm()
        #
        # return render(request,'upload.html',{'form':form})

    def get(self,request,**kwargs):
        form = UploadImageForm()
        return render(request,'upload.html',{'form':form})

class BrowsableApi(View):

    def get(self,request):
        form = BrowsableForm()
        return render(request,'upload.html',{'form':form})

    def post(self,request):
        form = BrowsableForm(request.POST,request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']
            card_type = form.cleaned_data['card_type']
            mode = form.cleaned_data['mode']
            bounds = None
            if mode == "bounded":
                bounds = form.cleaned_data['bounds']
                bounds = np.float32(json.loads(bounds))
            image = file_to_image(image_file)
            eh = ExtractorHandler(mode,card_type,image,bounds=bounds)
            data = eh.get_data()
            return JsonResponse(data)

        else:
            form = BrowsableForm()
            return render(request,'upload.html',{'form':form})
