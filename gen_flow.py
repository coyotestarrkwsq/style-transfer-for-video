import os
import cv2
import scipy.misc 
import numpy as np
import flow

WIDTH = 256
HEIGHT = 256
def get_img(src, img_size = False, grayscale = False):
    img = scipy.misc.imread(src, mode='RGB')
       
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))

    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
        
    if grayscale:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
 
    return img

def _get_files(img_dir, folder = False):
    files = list_files(img_dir, folder)
    return sorted([os.path.join(img_dir,x) for x in files])

def list_files(in_path, folder):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        if folder:
            files.extend(dirnames)
        else:
            files.extend(filenames)
        break

    return files

master_dir = "/home/wangsq/CS4243/DAVIS/"
master_dir = _get_files(master_dir, folder = True)

for subfolder in master_dir:
	img_files = _get_files(subfolder)
	count = 0
	filepath, _ = os.path.split(img_files[0])
	for i in range(0, len(img_files)-1):
		prev = get_img(img_files[i], (HEIGHT,WIDTH,3), grayscale = True).astype(np.float32)
		nxt = get_img(img_files[i+1], (HEIGHT,WIDTH,3), grayscale = True).astype(np.float32)
		forward_flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		flow.write_flo(filepath + "/flow/flow" + str(count) + ".flo", forward_flow)
		backward_flow = cv2.calcOpticalFlowFarneback(nxt, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		flow.write_flo(filepath + "/flow/flow" + str(count+1) + ".flo", backward_flow) 
		count+=2

		