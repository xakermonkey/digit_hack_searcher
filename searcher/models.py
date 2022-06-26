from django.db import models
from keras.models import load_model
import cv2
import numpy as np


# Create your models here.

def cell(x, ind):
    if ind == 0 and x > 3:
        return 3
    elif ind == 1 and x > 4:
        return 4
    elif ind == 2 and x > 2:
        return 2
    elif ind > 2 and x > 1:
        return 1
    if x % 1 >= 0.5:
        return round(x) + 1
    else:
        return round(x)


class Photo(models.Model):
    photo = models.ImageField(upload_to="image", verbose_name="Фотография", null=True, blank=True)
    tags = models.CharField(max_length=255, verbose_name="Закодированный список тегов", null=True, blank=True)

    class Meta:
        verbose_name = "Фотография"
        verbose_name_plural = "Фотографии"

    def __str__(self):
        return self.photo.url

    def save(self, *args, **kwargs):
        super(Photo, self).save(*args, **kwargs)
        if self.tags is None:
            model_tags = load_model("taggs.h5")
            img = cv2.imread("templates" + self.photo.url)
            img = cv2.resize(img, (640 // 2, 480 // 2))
            res = model_tags.predict(np.expand_dims(img, axis=0))[0]
            text = ";".join([str(cell(i, ind)) for ind, i in enumerate(res)])
            self.tags = text
        super(Photo, self).save(*args, **kwargs)



class TagsValue(models.Model):
    name = models.CharField(max_length=255, verbose_name="Значение тега")
    value = models.IntegerField(verbose_name="Численное значение")

    class Meta:
        verbose_name = "Значение тега"
        verbose_name_plural = "Значение тегов"

    def __str__(self):
        return f"{self.name} - {self.value}"


class Tags(models.Model):
    name = models.CharField(max_length=255, verbose_name="Название тега")
    values = models.ManyToManyField(TagsValue)

    class Meta:
        verbose_name = "Тег"
        verbose_name_plural = "Теги"

    def __str__(self):
        return self.name
