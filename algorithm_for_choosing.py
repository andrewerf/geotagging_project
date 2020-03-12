import numpy as np
import json
from sql_models import *
from tqdm import tqdm
from sklearn.cluster import OPTICS, MeanShift, KMeans


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


def split_by_labels(data, labels):
	count = int(np.max(labels))
	result = [[] for i in range(count + 1)]
	for i, x in enumerate(data):
		result[int(labels[i])].append(x)
	return result


def unite_by_labels(data):
	labels = list()
	result = list()
	for i, cluster in enumerate(data):
		for x in cluster:
			labels.append(i)
			result.append(x)

	return result, labels


def score_clustering(x, labels, proccess_Din=np.max):
	clusters = split_by_labels(x, labels)
	Din = list()
	avgs = list()
	for cluster in clusters:
		avgs.append(np.average(cluster))
		Din.append(np.var(cluster))

	Dout = np.var(avgs)
	return proccess_Din(Din) / Dout


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
	sights_req = Sight.select(Sight.id).limit(limits)
	sights = list()
	for row in sights_req:
		sights.append(convert_str_sight(row.sight_id))
	return sights


def get_all_descr_from_sight(sight_id):
	descrs_req = Descriptor.select().where(Descriptor.sight_id == sight_id)
	descrs = list()
	imgs_id = list()
	for row in descrs_req:
		descrs.append(convert_str_desc(row.descriptor))
		imgs_id.append(int(row.image_id))
	return (descrs, imgs_id)


def sum_descr(descr1, descr2):
	n = len(descr1)
	s_descr = [0] * n
	for i in range(n):
		s_descr[i] = descr1[i] + descr2[i]
	return s_descr


def get_similar_pos(data, metadata):
	sim_descrs = data[0]
	dist = data[1]
	imgs_id = data[2]
	coeffs = list()
	for i in range(len(dist)):
		if dist[i] == 0:
			avg_vec = sim_descrs[i]
			tdist = get_manhattan_dist(avg_vec, sim_descrs[0])
			img_id = imgs_id[0]
			for i in range(1, len(sim_descrs)):
				if get_manhattan_dist(sim_descrs[i], avg_vec) < tdist:
					tdist = get_manhattan_dist(sim_descrs[i], avg_vec)
					img_id = imgs_id[i]
			with open(metadata, "r") as read_file:
				mdata = json.load(read_file)
			for i in range(len(mdata)):
				if int(mdata[i]['id']) == img_id:
					return mdata[i]['loc']
		else:
			coeffs.append(1 / dist[i])
	sum_coeff = sum(coeffs)
	avg_vec = [0] * len(sim_descrs[0])
	for i in range(len(sim_descrs)):
		avg_vec = sum_descr(avg_vec, list(map(lambda x: x * coeffs[i], sim_descrs[i])))
	avg_vec = list(map(lambda x: x * (sum_coeff ** -1), avg_vec))
	tdist = get_manhattan_dist(avg_vec, sim_descrs[0])
	img_id = imgs_id[0]
	for i in range(1, len(sim_descrs)):
		if get_manhattan_dist(sim_descrs[i], avg_vec) < tdist:
			tdist = get_manhattan_dist(sim_descrs[i], avg_vec)
			img_id = imgs_id[i]
	with open(metadata, "r") as read_file:
		mdata = json.load(read_file)
	for i in range(len(mdata)):
		if int(mdata[i]['id']) == img_id:
			return mdata[i]['loc']


def get_avg_dist_pos(data, metadata):
	sim_descrs = data[0]
	dists = data[1]
	imgs_id = data[2]
	with open(metadata, "r") as read_file:
		mdata = json.load(read_file)
	coordinates = list()
	for j in range(len(imgs_id)):
		for i in range(len(mdata)):
			if int(mdata[i]['id']) == imgs_id[j]:
				coordinates.append([mdata[i]['loc']['lat'], mdata[i]['loc']['lng']])
				break
	if len(coordinates) == 1:
		return coordinates[0]
	coeffs = list()
	for i in range(len(dists)):
		if dists[i] == 0:
			avg_pos = dists[i]
			sim_pos = coordinates[0]
			sim_img_id = imgs_id[0]
			for i in range(1, len(coordinates)):
				if get_manhattan_dist(coordinates[i], avg_pos) < get_manhattan_dist(sim_pos, avg_pos):
					sim_pos = coordinates[i]
					sim_img_id = imgs_id[i]
			return avg_pos
		else:
			coeffs.append(1 / dists[i])
	avg_pos = [0] * len(coordinates[0])
	for i in range(len(coordinates)):
		avg_pos = sum_descr(avg_pos, list(map(lambda x: x * coeffs[i], coordinates[i])))
	coeff_sum = sum(coeffs)
	avg_pos = list(map(lambda x: (coeff_sum ** -1) * x, avg_pos))
	sim_pos = coordinates[0]
	sim_img_id = imgs_id[0]
	for i in range(1, len(coordinates)):
		if get_manhattan_dist(coordinates[i], avg_pos) < get_manhattan_dist(sim_pos, avg_pos):
			sim_pos = coordinates[i]
			sim_img_id = imgs_id[i]
	return avg_pos


def get_data_about_similar_descrs(sigth_id, c_desc, n):
	descrs, b_imgs_id = get_all_descr_from_sight(sigth_id)
	sim_descr = list()
	dist = list()
	imgs_id = list()
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


def get_similar_sight_id(descr, avgs_file):
	with open(avgs_file, "r") as read_file:
		avg_val_of_vecs = json.load(read_file)
	t_sight_id = next(iter(avg_val_of_vecs))
	dist = get_manhattan_dist(avg_val_of_vecs[t_sight_id][0], descr)
	for sight_id in avg_val_of_vecs.keys():
		if get_manhattan_dist(avg_val_of_vecs[sight_id][0], descr) < dist:
			dist = get_manhattan_dist(avg_val_of_vecs[sight_id][0], descr)
			t_sight_id = sight_id
	return t_sight_id


def get_avg_val_of_descrs(limits, fout):
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
	with open(fout, "w") as fp:
		json.dump(base_table, fp)


def sight_cluster(x, algo):
	clust = None
	if algo == 'optics':
		clust = OPTICS(metric='manhattan', n_jobs=-1, min_cluster_size=50)
	if algo == 'meanshift':
		clust = MeanShift(cluster_all=True, n_jobs=-1)
	if algo == 'kmeans':
		clust = KMeans(n_clusters=4, n_jobs=-1)

	return clust.fit_predict(x)
