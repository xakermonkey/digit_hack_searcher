import numpy as np
from urllib.parse import unquote
from django.shortcuts import render
from django.http import JsonResponse
import cv2
from keras.models import load_model
import os
from scipy import spatial
import pickle
import ruclip
import torch
import clip
import pandas as pd
from django.core.files import File
from .extract_features import *
from .models import *
from io import BytesIO
from django.db.models import Q

device = "cpu"
model_tags = load_model("taggs.h5")

ruClip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
model, preprocess = clip.load("ViT-B/32")

predictor = ruclip.Predictor(ruClip, processor, device, bs=8)

# generate_clip_features("templates/media/image", preprocess, model)
knn = get_knn()
image_latents = pk.load(open("ruclip_image_features.pkl", "rb"))


def main(request):
    tags = Tags.objects.all()
    # df = pd.read_csv("clear_dt.csv", index_col=0)
    # for i in Photo.objects.all():
    #     i.delete()
    # for i in range(len(df)):
    #     vector = np.array(df.iloc[i, 1:], dtype=float)
    #     photo = Photo.objects.create()
    #     photo.photo.name = "image/" + df.iloc[i, 0]
    #     photo.tags = ";".join(map(str, vector))
    #     photo.save()
    # Photo.objects.create(photo_path="templates/media/image/" + df.iloc[i, 0], tags=";".join(map(str, vector)))
    return render(request, 'index.html', {"tags": tags})


def find_by_text(request):
    # images = [Image.open("templates/media/image/" + file_name) for file_name in
    #           os.listdir("templates/media/image")[:5000] if
    #           not file_name.split(".")[-1] in bad_shape]
    with torch.no_grad():
        text_latents = predictor.get_text_latents([request.POST.get("text")])
        # image_latents = predictor.get_image_latents(images)
        # pk.dump(image_latents, open("ruclip_image_features.pkl", "wb"))
        logits_per_text = torch.matmul(text_latents, image_latents.t())
    res = logits_per_text.cpu().numpy()[0]
    q98 = np.percentile(res, 99)
    found_images = []
    file_names = os.listdir("templates/media/image")
    for x in range(len(res)):
        if res[x] >= q98:
            found_images.append("/media/image/" + file_names[x])
    return JsonResponse(status=200, data={"images": found_images})


def find_by_tags(request):
    tags = request.POST.getlist("tags[]")
    vector = np.zeros(Tags.objects.count())
    found_images = list()
    for tag in tags:
        indx, val = map(int, tag.split("_"))
        vector[indx - 1] = val
    for photo in Photo.objects.all():
        res = np.array(photo.tags.split(";"), dtype=float)
        if 1 - spatial.distance.cosine(res, vector) > 0.9:
            found_images.append(photo.photo.url)
    return JsonResponse(status=200, data={"images": found_images})


def upload_file(request):
    query_image_pillow = Image.open(BytesIO(request.FILES["find_img"].read()))
    query_image_features = get_features(query_image_pillow, preprocess, model)
    indices = knn.kneighbors(query_image_features, return_distance=False)
    found_images = []
    file_names = os.listdir("templates/media/image")
    for x in indices[0]:
        found_images.append("/media/image/" + file_names[x])
    return JsonResponse(status=200, data={
        "images": found_images})


def find_many_fields(request):
    with torch.no_grad():
        text_latents = predictor.get_text_latents([request.POST.get("text")])
        # image_latents = predictor.get_image_latents(images)
        # pk.dump(image_latents, open("ruclip_image_features.pkl", "wb"))
        logits_per_text = torch.matmul(text_latents, image_latents.t())
    res = logits_per_text.cpu().numpy()[0]
    q98 = np.percentile(res, 99)
    found_images = []
    file_names = os.listdir("templates/media/image")
    for x in range(len(res)):
        if res[x] >= q98:
            found_images.append("/media/image/" + file_names[x])
    tags = request.POST.getlist("tags[]")
    vector = np.zeros(Tags.objects.count())
    for tag in tags:
        indx, val = map(int, tag.split("_"))
        vector[indx - 1] = val
    result = list()
    for photo in Photo.objects.all():
        if unquote(photo.photo.url) in found_images:
            res = np.array(photo.tags.split(";"), dtype=float)
            if 1 - spatial.distance.cosine(res, vector) > 0.9:
                result.append(photo.photo.url)
    return JsonResponse(status=200, data={"images": result})
