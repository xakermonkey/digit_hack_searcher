
from .views import *
from django.urls import path


urlpatterns = [
    path("main", main, name='main'),
]
2