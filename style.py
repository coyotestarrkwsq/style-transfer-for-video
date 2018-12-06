import numpy as np
import cv2

import tensorflow as tf
import stylizing_net
import scipy.misc as misc
import glob 
import os

home = "/home/wangsq/CS4243/proj/" 
style = "d"
images = glob.glob(home + style + "/*png")

with tf.Graph().as_default():
	img = tf.placeholder(tf.float32, shape=(1,1080, 1920,3))
	transf_img = stylizing_net.net(img)

	saver = tf.train.Saver()


	with tf.Session() as sess:

		# Restore variables from disk.


		#saver.restore(sess, "/home/wangsq/CS4243/DAVIS_model/content_retained_model/la_muse_model/model.ckpt")
		#saver.restore(sess, "/home/wangsq/CS4243/DAVIS_model/starry_night_model/old/model.ckpt")
		#saver.restore(sess, "/home/wangsq/CS4243/DAVIS_model/models/scream_model/model.ckpt")
		#saver.restore(sess, "/home/wangsq/CS4243/DAVIS_model/models/ghjhg_model/model.ckpt")
		saver.restore(sess, "/home/wangsq/CS4243/DAVIS_model/models/dcvsdv_model/model.ckpt")

		print("Model restored.")


		for i in images:
			_, filename = os.path.split(i)
			frame = misc.imread(i)
			frame = np.array([frame])
			transf_frame = sess.run(transf_img, feed_dict = {img: frame})[0]
			misc.imsave(home + style + "/style_image/" + filename, transf_frame)
			print filename