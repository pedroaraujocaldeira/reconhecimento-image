from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from source.core import views


urlpatterns = [
    path('', views.upload, name='upload'),
    path('upload/', views.Home.as_view(), name='home'),
    # path('admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
