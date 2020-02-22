import json
import argparse

from geotagging_project.utils import Area, convert_accuracy

accuracy = 10000

parser = argparse.ArgumentParser()
parser.add_argument('--in', type=str, dest='fin')
parser.add_argument('--dir', type=str, dest='fout')

args = parser.parse_args()


def json_parse():
    pos = []

    with open(args.fin) as json_file:
        data = json.load(json_file)
        for p in data:
            pos.append((int(p['id']), float(p['loc']['lat']), float(p['loc']['lng'])))
    return pos


def group():
    data = json_parse()
    pos = []
    for p in data:
        area = Area((p[1], p[2]), accuracy)
        pos.append(area)
    areas = []
    i = 0
    for p in pos:
        if len(areas) == 0:
            areas.append([])
            areas[-1].append(p.pos)
            areas[-1].append([])
            areas[-1][-1].append(data[i][0])
        else:
            j = 0
            flag = False
            for buff in areas:
                if buff[0] in p:
                    flag = True
                    areas[j][1].append(data[i][0])
                j += 1
            if not flag:
                areas.append([])
                areas[-1].append(p.pos)
                areas[-1].append([])
                areas[-1][-1].append(data[i][0])
        i += 1
        print(i)
    with open(args.fout, "w") as fp:
        json.dump(areas, fp)


if __name__ == '__main__':
    # group()
    with open(args.fout) as json_data:
        data = json.load(json_data)
        print(len(data))

