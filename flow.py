import math
import cv2
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import tensorflow as tf

MOTION_BOUNDARIE_VALUE = 0.0

#flow1 is back, flow2 is forward
def get_flow_weights(flow1, flow2): 
  xSize = flow1.shape[1]
  ySize = flow1.shape[0]
  reliable = 255 * np.ones((ySize, xSize))

  size = xSize * ySize

  x_kernel = [[-0.5, -0.5, -0.5],[0., 0., 0.],[0.5, 0.5, 0.5]]
  x_kernel = np.array(x_kernel, np.float32)
  y_kernel = [[-0.5, 0., 0.5],[-0.5, 0., 0.5],[-0.5, 0., 0.5]]
  y_kernel = np.array(y_kernel, np.float32)
  
  flow_x_dx = cv2.filter2D(flow1[:,:,0],-1,x_kernel)
  flow_x_dy = cv2.filter2D(flow1[:,:,0],-1,y_kernel)
  dx = np.stack((flow_x_dx, flow_x_dy), axis = -1)

  flow_y_dx = cv2.filter2D(flow1[:,:,0],-1,x_kernel)
  flow_y_dy = cv2.filter2D(flow1[:,:,0],-1,y_kernel)
  dy = np.stack((flow_y_dx, flow_y_dy), axis = -1)

  motionEdge = np.zeros((ySize,xSize))

  for i in range(ySize):
    for j in range(xSize): 
      motionEdge[i,j] += dy[i,j,0]*dy[i,j,0]
      motionEdge[i,j] += dy[i,j,1]*dy[i,j,1]
      motionEdge[i,j] += dx[i,j,0]*dx[i,j,0]
      motionEdge[i,j] += dx[i,j,1]*dx[i,j,1]
      

  for ax in range(xSize):
    for ay in range(ySize): 
      bx = ax + flow1[ay, ax, 0]
      by = ay + flow1[ay, ax, 1]    

      x1 = int(bx)
      y1 = int(by)
      x2 = x1 + 1
      y2 = y1 + 1
      
      if x1 < 0 or x2 >= xSize or y1 < 0 or y2 >= ySize:
        reliable[ay, ax] = 0.0
        continue 
      
      alphaX = bx - x1 
      alphaY = by - y1

      a = (1.0-alphaX) * flow2[y1, x1, 0] + alphaX * flow2[y1, x2, 0]
      b = (1.0-alphaX) * flow2[y2, x1, 0] + alphaX * flow2[y2, x2, 0]
      
      u = (1.0 - alphaY) * a + alphaY * b
      
      a = (1.0-alphaX) * flow2[y1, x1, 1] + alphaX * flow2[y1, x2, 1]
      b = (1.0-alphaX) * flow2[y2, x1, 1] + alphaX * flow2[y2, x2, 1]
      
      v = (1.0 - alphaY) * a + alphaY * b
      cx = bx + u
      cy = by + v
      u2 = flow1[ay,ax,0]
      v2 = flow1[ay,ax,1]
      
      if ((cx-ax) * (cx-ax) + (cy-ay) * (cy-ay)) >= 0.01 * (u2*u2 + v2*v2 + u*u + v*v) + 0.5: 
        # Set to a negative value so that when smoothing is applied the smoothing goes "to the outside".
        # Afterwards, we clip values below 0.
        reliable[ay, ax] = -255.0
        continue
      
      if motionEdge[ay, ax] > 0.01 * (u2*u2 + v2*v2) + 0.002: 
        reliable[ay, ax] = MOTION_BOUNDARIE_VALUE
        continue
      
  #need to apply smoothing to reliable mat
  reliable = cv2.GaussianBlur(reliable,(3,3),0)
  reliable = np.clip(reliable, 0.0, 255.0)    
  return reliable  

def write_flo(filename, flow):

    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width, channel = flow.shape
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    #empty_map = np.zeros((height, width), dtype=np.float32)
    #data = np.dstack((flow, empty_map))
    data=flow
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data.tofile(f)
    f.close()

def read_flo(file):
  with open(file) as f:
    magic = np.fromfile(f, np.float32, count=1)
    if 202021.25 != magic:
      print 'Magic number incorrect. Invalid .flo file'
    else:
      w = np.fromfile(f, np.int32, count=1)[0]
      h = np.fromfile(f, np.int32, count=1)[0]
      print 'Reading %d x %d flo file' % (h, w)
      data = np.fromfile(f, np.float32, count=2*w*h)
      # Reshape data into 3D array (columns, rows, bands)
      data2D = np.resize(data, (h, w, 2))
  return data2D




def get_temporal_loss(nxt, warped_prev, c):
  D = tf.size(nxt, out_type = tf.float32)
  loss = (1. / D) * tf.reduce_sum(tf.multiply(c, tf.squared_difference(nxt, warped_prev)))
  loss = tf.cast(loss, tf.float32)
  return loss



def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res



'''
forw = read_flo("frame_0001.flo")

prev = misc.imread("frame_0001.png")
nxt = warp_flow(prev, forw)
misc.imsave("frame_0002.png", nxt)
'''