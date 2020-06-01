from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.generic import TemplateView
from source.core.machine.model import open_model_64x3
from .forms import ImageForm

IMAGE_FILE_TYPES = ['png', 'jpg', 'jpeg']


class Home(TemplateView):
    template_name = 'home.html'


def upload(request):
    context = {}
    form = ImageForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        uploaded_file = request.FILES['image']
        file_type = uploaded_file.name.split('.')[1]
        if file_type not in IMAGE_FILE_TYPES:
            return render(request, 'upload.html', {'error_message': 'O arquivo de imagem deve ser PNG, JPG ou JPEG'})

        fs = FileSystemStorage()
        name = fs.save('image.' + file_type, uploaded_file)
        context['url'] = fs.url(name)
        context['classification'] = open_model_64x3(context['url'][1:])

    return render(request, 'upload.html', context)


