from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from .forms import LogoUploadForm

# Create your views here.
model = load_model('fake_logo_detector.h5')

def index(request):
    form = LogoUploadForm()
    return render(request, 'detection/index.html', {'form': form})

def predict(request):
    if request.method == 'POST':
        form = LogoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['logo']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_url = fs.url(filename)

            img = image.load_img(fs.path(filename), target_size=(150, 150))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.0

            prediction = model.predict(img_tensor)
            result = 'Fake' if prediction < 0.5 else 'Real'

            return render(request, 'detection/result.html', {'result': result, 'file_url': file_url})
    return render(request, 'detection/index.html', {'form': form})
#