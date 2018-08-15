
import os
import glob
from os.path import join
import json
import cv2
import numpy as np

global counter
global sum

counter = 0
sum = 0

def check_and_make_dir(path):
	if not os.path.exists(path):
		 os.mkdir(path)

def load_batch_from_data(names, path, batch_size, img_ch, img_cols, img_rows, train_start = None, train_end = None):
	global counter
	global sum
	save_img = False
	for i in range(len(names)):
		# lottery
		# i = np.random.randint(0, len(names))

		# get the lucky one
		img_name = names[i]
		if counter % 100 == 0:
			print ("counter: ", counter)
			print ("sum: ", sum)

		print ("img_name: ", img_name)

		counter += 1
		# directory
		dir = img_name[:5]
		if int(dir) >= 0:
			# frame name
			frame = img_name[6:]

			# index of the frame into a sequence
			idx = int(frame[:-4])

			# open json files
			face_file = open(join(path, dir, "appleFace.json"))
			left_file = open(join(path, dir, "appleLeftEye.json"))
			right_file = open(join(path, dir, "appleRightEye.json"))
			dot_file = open(join(path, dir, "dotInfo.json"))
			grid_file = open(join(path, dir, "faceGrid.json"))

			# load json content
			face_json = json.load(face_file)
			left_json = json.load(left_file)
			right_json = json.load(right_file)
			dot_json = json.load(dot_file)
			grid_json = json.load(grid_file)

			# open image
			img = cv2.imread(join(path, dir, "frames", frame))

			# if image is null, skip
			if img is None:
				# print("Error opening image: {}".format(join(path, dir, "frames", frame)))
				continue

			# if coordinates are negatives, skip (a lot of negative coords!)
			if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
				int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
				int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
				# print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
				continue

			# get face
			tl_x_face = int(face_json["X"][idx])
			tl_y_face = int(face_json["Y"][idx])
			w = int(face_json["W"][idx])
			h = int(face_json["H"][idx])
			br_x = tl_x_face + w
			br_y = tl_y_face + h
			face = img[tl_y_face:br_y, tl_x_face:br_x]

			# get left eye
			tl_x = tl_x_face + int(left_json["X"][idx])
			tl_y = tl_y_face + int(left_json["Y"][idx])
			w = int(left_json["W"][idx])
			h = int(left_json["H"][idx])
			br_x = tl_x + w
			br_y = tl_y + h
			left_eye = img[tl_y:br_y, tl_x:br_x]

			# get right eye
			tl_x = tl_x_face + int(right_json["X"][idx])
			tl_y = tl_y_face + int(right_json["Y"][idx])
			w = int(right_json["W"][idx])
			h = int(right_json["H"][idx])
			br_x = tl_x + w
			br_y = tl_y + h
			right_eye = img[tl_y:br_y, tl_x:br_x]

			# # get face grid (in ch, cols, rows convention)
			# # face_grid = np.zeros(shape=(25, 25, 1), dtype=np.float32)
			# face_grid = np.zeros(shape=(25, 25), dtype=np.float32)
			# tl_x = int(grid_json["X"][idx])
			# tl_y = int(grid_json["Y"][idx])
			# w = int(grid_json["W"][idx])
			# h = int(grid_json["H"][idx])
			# br_x = tl_x + w
			# br_y = tl_y + h
			#
			# # print ("face_grid: ", face_grid.shape)
			# # face_grid[0, tl_y:br_y, tl_x:br_x] = 1
			# face_grid[tl_y:br_y, tl_x:br_x] = 1
			# # face_grid = face_grid.flatten()

			# # get labels
			# y_x = dot_json["XCam"][idx]
			# y_y = dot_json["YCam"][idx]

			for folder in ["/appleFace/", "/appleLeftEye/", "/appleRightEye/"]:
				p = join(path, dir) + folder
				check_and_make_dir(p)

			cv2.imwrite(join(path, dir, "appleFace", frame), face)
			cv2.imwrite(join(path, dir, "appleRightEye", frame), right_eye)
			cv2.imwrite(join(path, dir, "appleLeftEye", frame), left_eye)
			# cv2.imwrite("images/image.png", img)
			# raise "debug"

# create a list of all names of images in the dataset
def load_data_names(path):

	seq_list = []
	seqs = sorted(glob.glob(join(path, "0*")))

	for seq in seqs:
		file = open(seq, "r")
		content = file.read().splitlines()
		for line in content:
			seq_list.append(line)

	return seq_list




img_cols = 64
img_rows = 64
img_ch = 3

dataset_path = "..\..\Eye-Tracking-for-Everyone-master\Eye-Tracking-for-Everyone-master\GazeCapture"
train_path = dataset_path + '\ '.strip() + "train"
val_path = dataset_path + '\ '.strip() + "validation"
test_path = dataset_path + '\ '.strip() + "test"


train_names = load_data_names(train_path)
val_names = load_data_names(val_path)
test_names = load_data_names(test_path)

print ("train_names: ", len(train_names))
print ("val_names: ", len(val_names))
print ("test_names: ", len(test_names))
sum = len(train_names) + len(val_names) + len(test_names)
print ("sum: ", sum)

train_data = load_batch_from_data(train_names , dataset_path, None, img_ch, img_cols, img_rows)

val_data = load_batch_from_data(val_names, dataset_path, None, img_ch, img_cols, img_rows)

test_data = load_batch_from_data(test_names, dataset_path, None, img_ch, img_cols, img_rows)
