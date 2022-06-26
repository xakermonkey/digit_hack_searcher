
from .views import *
from django.urls import path


urlpatterns = [
    path("main", main, name='main'),
    path("upload_file", upload_file, name='addfile'),
    path("find_text", find_by_text, name='find_text'),
    path("find_image", find_by_text, name='find_text'),
    path("find_tags", find_by_tags, name='find_tags'),
    path("find_all", find_many_fields, name='find_all'),
]