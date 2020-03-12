import numpy as np
import csv
import argparse
import json
from tqdm import tqdm
from geotagging_project.sql_models import db_handler, Sight, Descriptor
from geotagging_project.algorithm_for_choosing import convert_str_sight, get_all_descr_from_sight, score_clustering, sum_descr
from sklearn.cluster import OPTICS

number_of_classes = 50


def convert_str_desc(s):
    desc = []
    buff = ''
    for i in range(len(s)):
        if s[i] == ',':
            desc.append(float(buff))
            buff = ''
        else:
            buff += s[i]
    desc.append(float(buff))
    return desc


def get_base_of_descrs_from_csv(fin, limit=-1):
    descrs = []
    with open(fin) as file:
        reader = csv.reader(file)
        j = 0
        for row in reader:
            descrs.append(list(map(lambda x: float(x), row)))
            if j == limit:
                break
            j += 1
    return descrs


def get_base_of_descrs_from_db():
    descrs = Descriptor.select()
    descrs_data = []
    for descr in descrs:
        descrs_data.append(convert_str_desc(descr.descriptor))
    return descrs_data


def avg_val_of_vec(vec1, vec2):
    n = len(vec1)
    vec3 = []
    for i in range(n):
        vec3.append((vec1[i] + vec2[i]) / 2)
    return vec3


def get_unic_sigths_id(limits=-1):
    if limits != -1:
        sights_req = Sight.select(Sight.id).limit(limits)
    else:
        sights_req = Sight.select(Sight.id)
    sights = list()
    for row in sights_req:
        sights.append(convert_str_sight(row.id))
    return sights


def get_data(descrs_path, limit=-1):
    descrs = list()
    with open(descrs_path, "r") as f_obj:
        reader = csv.reader(f_obj)
        i = 0
        for row in tqdm(reader):
            descrs.append(list(map(lambda x: float(x), row)))
            if i == limit:
                break
            i += 1
    return descrs


def get_unic_labels(labels):
    unic_labels = list()
    for val in labels:
        if not val in unic_labels:
            unic_labels.append(val)
    unic_labels.sort()
    unic_labels.pop(0)
    return unic_labels


def get_all_descr_from_label(data, labels, label):
    descrs = list()
    for i, val in enumerate(data):
        if labels[i] == label:
            descrs.append(val)
    return descrs


def get_avg_descr_for_all_clust(data, labels):
    avg_descrs = dict()
    for label in tqdm(get_unic_labels(labels)):
        descrs = get_all_descr_from_label(data, labels, label)
        n = len(descrs[0])
        avg_descr = [0] * n
        for descr in descrs:
            avg_descr = sum_descr(avg_descr, descr)
        avg_descr = list(map(lambda x: x * (n ** -1), avg_descr))
        avg_descrs[str(label)] = avg_descr
    return avg_descrs


def descrs_cluster(data):
    clust = OPTICS(metric='manhattan', n_jobs=-1, min_cluster_size=200)
    return clust.fit_predict(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', type=str, dest='descrs_path')
    parser.add_argument('--out', type=str, dest='json_path')
    args = parser.parse_args()

    data = get_data(args.descrs_path)
    labels = descrs_cluster(data)
    print(score_clustering(data, labels, np.average))
    with open(args.json_path, "w") as write_file:
        json.dump(get_avg_descr_for_all_clust(data, labels), write_file)

