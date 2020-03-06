import numpy as np
import argparse
import csv
from geotagging_project.kmedoids import *
from sklearn.metrics.pairwise import pairwise_distances
from geotagging_project.sql_models import *
from tqdm import  tqdm

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, dest='fout')
    parser.add_argument('--in', type=str, dest='fin')

    args = parser.parse_args()

    data = np.array(get_base_of_descrs_from_csv(args.fin, limit=100))

    D = pairwise_distances(data, metric='manhattan')

    M, C = kMedoids(D, number_of_classes)

    # print('')
    # print('clustering result:')
    avg_desc_data = []
    for label in tqdm(C):
        avg_desc = data[C[label][0]]
        for point_idx in C[label]:
            # print('Label [{0}]:\t\t{1}'.format(label, data[point_idx]))
            avg_desc = avg_val_of_vec(avg_desc, data[point_idx])
        avg_desc_data.append(avg_desc)

    with open(args.fout, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(avg_desc_data)
