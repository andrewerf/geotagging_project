import numpy as np
import argparse
import csv
from geotagging_project.kmedoids import *
from sklearn.metrics.pairwise import pairwise_distances
from geotagging_project.sql_models import *

number_of_classes = 5000


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


def get_base_of_decrs():
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

    args = parser.parse_args()

    data = np.array(get_base_of_decrs())

    D = pairwise_distances(data, metric='manhattan')

    M, C = kMedoids(D, number_of_classes)

    print('')
    print('clustering result:')
    avg_desc_data = []
    for label in C:
        avg_desc = data[C[label][0]]
        for point_idx in C[label]:
            print('Label [{0}]:\t\t{1}'.format(label, data[point_idx]))
            avg_desc = avg_val_of_vec(avg_desc, data[point_idx])
        avg_desc_data.append(avg_desc)

    with open(args.fout, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(avg_desc_data)
