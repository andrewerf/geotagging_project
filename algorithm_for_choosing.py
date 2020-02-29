import numpy as np
import argparse
import json
from geotagging_project.test_veiw import show_images, show_image
from geotagging_project.sql_models import *
from geotagging_project.mobilenetv1_encoder import Encoder, load_img
from geotagging_project.classifier import Classifier, classes_count
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--in', type=str, dest='fin')
parser.add_argument('--out', type=str, dest='fout')
parser.add_argument('--metadata', type=str, dest='metadata')
parser.add_argument('--classes', type=str, dest='fclasses')
args = parser.parse_args()


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


def avg_val_of_vec(vec1, vec2):
	n = min(len(vec1), len(vec2))
	vec3 = []
	for i in range(n):
		vec3.append((vec1[i] + vec2[i]) / 2)
	return vec3


def convert_str_sight(s):
	return int(str(s))


def get_manhattan_dist(descr1, descr2):
	dist = 0
	for i in range(min(len(descr1), len(descr2))):
		dist += abs(descr1[i] - descr2[i])
	return dist


def get_sq_of_euclid_dist(descr1, descr2):
	dist = 0
	for i in range(min(len(descr1), len(descr2))):
		dist += (descr1[i] - descr2[i]) ** 2
	return dist


def get_nearest_descr_from_all(limit, cur_descr, dist_metric):
	if limit == -1:
		descrs = Descriptor.select()
	else:
		descrs = Descriptor.select().limit(limit)
	if dist_metric == 'manhattan':
		dist = get_manhattan_dist(convert_str_desc(descrs[0].descriptor), cur_descr)
		nearest_descr = convert_str_desc(descrs[0].descriptor)
		sight = convert_str_sight(descrs[0].sight_id)
		for row in descrs:
			desc = convert_str_desc(row.descriptor)
			if get_manhattan_dist(desc, cur_descr) < dist:
				dist = get_manhattan_dist(desc, cur_descr)
				nearest_descr = desc
				sight = convert_str_sight(row.sight_id)
	elif dist_metric == 'square_of_euclid':
		dist = get_sq_of_euclid_dist(convert_str_desc(descrs[0].descriptor), cur_descr)
		nearest_descr = convert_str_desc(descrs[0].descriptor)
		sight = convert_str_sight(descrs[0].sight_id)
		for row in descrs:
			desc = convert_str_desc(row.descriptor)
			if get_sq_of_euclid_dist(desc, cur_descr) < dist:
				dist = get_sq_of_euclid_dist(desc, cur_descr)
				nearest_descr = desc
				sight = convert_str_sight(row.sight_id)
	return (nearest_descr, sight)


def get_unic_sigth_id(limits):
	sights_req = Descriptor.select().group_by(Descriptor.sight_id).having(fn.COUNT(Descriptor.id) == 1).limit(limits)
	sights = []
	for row in sights_req:
		sights.append(convert_str_sight(row.sight_id))
	return sights


def get_all_descr_from_sight(sight_id):
	descrs_req = Descriptor.select().where(Descriptor.sight_id == sight_id)
	descrs = []
	imgs_id = []
	for row in descrs_req:
		descrs.append(convert_str_desc(row.descriptor))
		imgs_id.append(int(row.image_id))
	return (descrs, imgs_id)


def get_similar_pos(data):
	sim_descrs = data[0]
	dist = data[1]
	imgs_id = data[2]
	dist_mn = min(dist)
	coeffs = []
	for i in range(len(dist)):
		if dist[i] == 0:
			coeffs.append(1.0)
		else:
			coeffs.append(dist_mn / dist[i])
	avg_vec = sim_descrs[0]
	for i in range(1, len(sim_descrs)):
		avg_vec = avg_val_of_vec(list(map(lambda x: x * coeffs[i - 1], avg_vec)),
								 list(map(lambda x: x * coeffs[i], sim_descrs[i])))
	tdist = get_manhattan_dist(avg_vec, sim_descrs[0])
	for i in range(1, len(sim_descrs)):
		if get_manhattan_dist(sim_descrs[i], avg_vec) < tdist:
			tdist = get_manhattan_dist(sim_descrs[i], avg_vec)
			img_id = imgs_id[i]
	show_images(str(img_id), '/media/qunity/Workspace/Python_projects/NeuralNetworks/Images')
	with open(args.metadata, "r") as read_file:
		mdata = json.load(read_file)
	for i in range(len(mdata)):
		if int(mdata[i]['id']) == img_id:
			return mdata[i]['loc']


# def get_min_dist_pos(data, sight_id):
# 	sim_descrs = data[0]
# 	dist = data[0]
# 	imgs_id = get_all_descr_from_sight(sight_id)[1]
# 	m_dist = min(dist)
# 	for i in range(len(sim_descrs)):
# 		if


def get_similar_descrs(sigth_id, c_desc, n):
	descrs, b_imgs_id = get_all_descr_from_sight(sigth_id)
	sim_descr = []
	dist = []
	imgs_id = []
	j = 0
	for desc in descrs:
		if len(sim_descr) < n:
			sim_descr.append(desc)
			dist.append(get_manhattan_dist(desc, c_desc))
			imgs_id.append(b_imgs_id[j])
		else:
			for i in range(len(sim_descr)):
				if get_manhattan_dist(desc, c_desc) < dist[i]:
					dist[i] = get_manhattan_dist(desc, c_desc)
					sim_descr[i] = desc
					imgs_id[i] = b_imgs_id[j]
					break
		j += 1
	return (sim_descr, dist, imgs_id)


def get_similar_sight_id(descr):
	with open(args.fin, "r") as read_file:
		avg_val_of_vecs = json.load(read_file)
	t_sight_id = next(iter(avg_val_of_vecs))
	dist = get_manhattan_dist(avg_val_of_vecs[t_sight_id][0], descr)
	for sight_id in avg_val_of_vecs.keys():
		if get_manhattan_dist(avg_val_of_vecs[sight_id][0], descr) < dist:
			dist = get_manhattan_dist(avg_val_of_vecs[sight_id][0], descr)
			t_sight_id = sight_id
	return t_sight_id


def get_avg_val_of_descrs(limits):
	if limits == -1:
		descrs_req = Descriptor.select()
	else:
		descrs_req = Descriptor.select().limit(limits)
	base_table = {}
	for row in tqdm(descrs_req):
		desc = convert_str_desc(row.descriptor)
		sight_id = convert_str_sight(row.sight_id)
		if not sight_id in base_table:
			base_table[sight_id] = []
			base_table[sight_id].append(desc)
		else:
			avg_desc = avg_val_of_vec(base_table[sight_id][0], desc)
			base_table[sight_id].pop(0)
			base_table[sight_id].append(avg_desc)
	with open(args.fout, "w") as fp:
		json.dump(base_table, fp)


def main():
	img_path = 'imgs/florence3.jpg'
	img = load_img(img_path)
	descr = Encoder().get_descriptor(img)

	Class_model = Classifier(classes_count, descr.shape, weights_file='stuff/weights.h5')

	pred = np.argmax(Class_model.predict(descr))
	accuacy = Class_model.predict(descr)
	with open(args.fclasses, "r") as read_file:
		classes = json.load(read_file)
	sight_id = classes[pred]
	show_image(img_path)
	print(get_similar_pos(get_similar_descrs(sight_id, descr, 3)))


if __name__ == '__main__':
	main()
