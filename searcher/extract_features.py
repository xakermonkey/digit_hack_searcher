import os
import torch
import clip
from os import listdir
from os.path import splitext
import json
from PIL import Image
import pickle as pk
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np

bad_shape = ["HEIC", "ORF", "ARW", "DS_Store"]


def get_features(image, preprocess, model):
    image = preprocess(image).unsqueeze(0).to('cpu')
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()


def generate_clip_features(path, preprocess, model):
    all_image_features = []
    image_filenames = listdir(path)
    try:
        all_image_features = pk.load(open("image_features.pkl", "rb"))
    except (OSError, IOError) as e:
        print("file_not_found")

    def exists_in_all_image_features(image_id):
        for image in all_image_features:
            if image['image_id'] == image_id:
                # print("skipping "+ str(image_id))
                return True
        return False

    def exists_in_image_folder(image_id):
        for filename in image_filenames:
            if splitext(filename)[0] == image_id:
                return True
        return False

    def sync_clip_image_features():
        for_deletion = []
        for i in range(len(all_image_features)):
            if not exists_in_image_folder(all_image_features[i]['image_id']):
                print("deleting " + str(all_image_features[i]['image_id']))
                for_deletion.append(i)
        for i in reversed(for_deletion):
            del all_image_features[i]

    sync_clip_image_features()
    for image_filename in tqdm(image_filenames):
        if not image_filename.split(".")[-1] in bad_shape:
            image_id = splitext(image_filename)[0]
            if exists_in_all_image_features(image_id):
                continue
            image = Image.open(path + "/" + image_filename)
            image_features = get_features(image, preprocess, model)
            all_image_features.append({'image_id': image_id, 'features': image_features})
    pk.dump(all_image_features, open("image_features.pkl", "wb"))


def get_knn():
    image_features = pk.load(open("image_features.pkl", "rb"))
    features = []
    for image in image_features:
        features.append(np.array(image['features']))
    features = np.array(features)
    features = np.squeeze(features)
    knn = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean') # 'cosine' 'euclidean'
    knn.fit(features)
    return knn


def generate_ruclip_features(path, model):
    all_image_features = []
    image_filenames = listdir(path)
    try:
        all_image_features = pk.load(open("ruclip_image_features.pkl", "rb"))
    except (OSError, IOError) as e:
        print("file_not_found")

    def exists_in_all_image_features(image_id):
        for image in all_image_features:
            if image['image_id'] == image_id:
                # print("skipping "+ str(image_id))
                return True
        return False

    def exists_in_image_folder(image_id):
        for filename in image_filenames:
            if splitext(filename)[0] == image_id:
                return True
        return False

    def sync_clip_image_features():
        for_deletion = []
        for i in range(len(all_image_features)):
            if not exists_in_image_folder(all_image_features[i]['image_id']):
                print("deleting " + str(all_image_features[i]['image_id']))
                for_deletion.append(i)
        for i in reversed(for_deletion):
            del all_image_features[i]

    sync_clip_image_features()
    for image_filename in tqdm(image_filenames):
        if not image_filename.split(".")[-1] in bad_shape:
            image_id = splitext(image_filename)[0]
            if exists_in_all_image_features(image_id):
                continue
            image = Image.open(path + "/" + image_filename)
            image_features = model.get_image_latents([image])
            all_image_features.append({'image_id': image_id, 'features': image_features})
    pk.dump(all_image_features, open("ruclip_image_features.pkl", "wb"))
