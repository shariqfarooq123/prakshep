from django import forms

class UploadImageForm(forms.Form):
    image = forms.ImageField()

class BrowsableForm(forms.Form):
    card_type = forms.ChoiceField(choices=[('pan','pan'),('aadhaar','aadhaar')])
    mode = forms.ChoiceField(choices=[('fit','fit'),('raw','raw'),('bounded','bounded')])
    bounds = forms.CharField(required=False)
    image = forms.ImageField()
