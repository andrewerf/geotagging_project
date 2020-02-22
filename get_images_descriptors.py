import os
import csv
import argparse

from mobilenetv1_encoder import Encoder, load_img



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--indir', type=str, dest='din')
	parser.add_argument('--out', type=str, dest='fout')

	args = parser.parse_args()
	fout = open(args.fout, 'w', newline='')
	writer = csv.writer(fout)
	en = Encoder()

	for root, dirs, files in os.walk(args.din):

		for fname in files:
			if '.jpg' in fname or '.jpeg' in fname:
				img = load_img(os.path.join(root, fname))
				descrs = en.get_descriptor(img)
				writer.writerow([os.path.splitext(fname)[0]] + descrs.tolist())


