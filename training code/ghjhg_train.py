import functools
import loss_net, time
import tensorflow as tf, numpy as np
import stylizing_net
import cv2
import flow
import os
import scipy

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TEMPORAL_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
VGG_PATH = '/home/wangsq/CS4243/imagenet-vgg-verydeep-19.mat'
HEIGHT = 256
WIDTH = 256

def _get_files(img_dir, folder = False):
    files = list_files(img_dir, folder)
    return sorted([os.path.join(img_dir,x) for x in files])

    


def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)



def get_img(src, img_size = False, grayscale = False):
    img = scipy.misc.imread(src, mode='RGB')
       
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))

    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
        
    if grayscale:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
 
    return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path, folder):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        if folder:
            files.extend(dirnames)
        else:
            files.extend(filenames)
        break

    return files


    
    
STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_2', 'relu4_2')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'


# np arr, np arr
def optimize(save_path, content_targets, style_target, content_weight ,style_weight, temporal_weight,
             tv_weight, loss_net_path, log_dir, epochs=NUM_EPOCHS, print_iterations=50,             
             learning_rate=1e-3, debug=False):

    batch_size=2
    style_features = {}

    batch_shape = (batch_size,HEIGHT,WIDTH,3)    
    flow_shape = (HEIGHT, WIDTH, 2)
      
    style_shape = (1,) + style_target.shape
    
    print style_shape
    
    # precompute style features
    with tf.Graph().as_default(), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = loss_net.preprocess(style_image)
        net = loss_net.net(loss_net_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(),  tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        
        X_pre = loss_net.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = loss_net.net(loss_net_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        preds = stylizing_net.net(X_content/255.0)

        preds_pre = loss_net.preprocess(preds)
        net = loss_net.net(loss_net_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )
        tf.summary.scalar('conten_loss', content_loss)


        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size
        tf.summary.scalar('style_loss', style_loss)


        #compute temporal loss
        X_flow_weights = tf.placeholder(tf.float32, shape=(HEIGHT,WIDTH,3), name="X_flow_weights")
        X_warped_image = tf.placeholder(tf.float32, shape=(HEIGHT,WIDTH,3), name="X_warped_image") 
                  
        nxt_transf_img = preds[1]
        
        temporal_loss = temporal_weight * flow.get_temporal_loss(nxt_transf_img, 
                                                            X_warped_image, X_flow_weights)
        
        tf.summary.scalar('temporal_loss', temporal_loss)


        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
        tf.summary.scalar('tv_loss', tv_loss)
        
        
        X_step = tf.placeholder(tf.int32)
        
        def f1(): return content_loss + style_loss + tv_loss
        def f2(): return content_loss + style_loss + temporal_loss + tv_loss
        loss = tf.cond(tf.greater(X_step, 0), f2, f1) 
        tf.summary.scalar('loss', loss)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        
        saver = tf.train.Saver()

        # overall loss

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        
        #saver.restore(sess, "/home/wangsq/CS4243/DAVIS_model/models/ghjhg_model/model.ckpt")

        num_videos = len(content_targets)    

        global_step = 0
        warped_image = np.zeros((HEIGHT, WIDTH, 3))
        
        first_folder = content_targets[0]
        first_frame = _get_files(first_folder)[0]

        first_filepath, _ = os.path.split(first_frame)  
        first_flowfile = first_filepath + '/flow/flow0.flo'
        first_flow = flow.read_flo(first_flowfile) 

        for epoch in range(epochs):
            
            for k in range(num_videos):

                subfolder = content_targets[k]
                content_target = _get_files(subfolder)
                num_frames = len(content_target)
                
                filepath, _ = os.path.split(content_target[0])  
                flowfilepath = filepath + "/flow"
                
                if k < num_videos -1:
                    nxt_subfolder = content_targets[k+1]
                    nxt_first_frame = _get_files(nxt_subfolder)[0]
                    nxt_filepath, _ = os.path.split(nxt_first_frame)  
                    nxt_flowfile = nxt_filepath + '/flow/flow0.flo'
                    nxt_flow = flow.read_flo(nxt_flowfile) 
                
             
                
                if global_step == 0:
                    start = -1
                else:
                    start = 0

                for i in range(start, num_frames-1):
                    global_step += 1
                    
                    print "i = " + str(i)

                    X_batch = np.zeros(batch_shape, dtype=np.float32)                    
                    
                    if i == -1:     
                        X_batch[1] = get_img(content_target[i+1], (HEIGHT,WIDTH,3)).astype(np.float32)
                    else:  
                        for j in range(2):
                            X_batch[j] = get_img(content_target[i+j], (HEIGHT,WIDTH,3)).astype(np.float32)
                               
                    print 'using precomputed flow'
                    
                    flowfile = flowfilepath + "/flow"

                    if i > -1:
                        forward_flow = flow.read_flo(flowfile + str(i*2) + ".flo")
                        print(flowfile + str(i*2) + ".flo")
                        backward_flow = flow.read_flo(flowfile + str(i*2+1) + ".flo")
                        print(flowfile + str(i*2+1) + ".flo")

                        flow_weights = flow.get_flow_weights(backward_flow, forward_flow)
                        flow_weights = np.stack((flow_weights,flow_weights,flow_weights), axis=-1)
                    else:
                        flow_weights = np.zeros((HEIGHT,WIDTH,3), dtype=np.float32) 
                    
                    if i == -1:
                        warp_flow = flow.read_flo(flowfile + str(0)+".flo")
                        print(flowfile + str(0)+".flo")

                    if i > -1 and i < num_frames - 2:
                        warp_flow = flow.read_flo(flowfile + str(i*2+2)+".flo")
                        print(flowfile + str(i*2+2)+".flo")

                    if i == num_frames - 2:
                        warp_flow = nxt_flow
                        print("reading flow first flow file from next folder")

                    if i == num_frames - 2 and k == num_videos - 1:
                        warp_flow = first_flow
                        print("going into next epoch, reading first flow file of first folder")


                          
                    
                    
                    to_get = [style_loss, content_loss, temporal_loss, tv_loss, loss]
                    
                    feed_dict = {
                        X_content : X_batch,
                        X_flow_weights : flow_weights,
                        X_warped_image : warped_image,
                        X_step : global_step
                    }
    
                    to_get = [style_loss, content_loss, temporal_loss, tv_loss, loss]
                    _, to_print = sess.run([train_step, to_get], feed_dict = feed_dict)
                    
                         
                        
                    
                    if i == num_frames -2:
                        X_batch[1] = get_img(nxt_first_frame, (HEIGHT,WIDTH,3)).astype(np.float32)
                    
                    if k == num_videos - 1 and i == num_frames - 2 :
                        X_batch[1] = get_img(first_frame, (HEIGHT,WIDTH,3)).astype(np.float32)

                    warp_feed_dict = {
                        X_content : X_batch,
                        X_flow_weights : flow_weights,
                        X_warped_image : warped_image,
                        X_step : global_step
                    }

                    transf_images = sess.run(preds, feed_dict = warp_feed_dict)
                    image_to_warp = transf_images[1] 
                    warped_image = flow.warp_flow(image_to_warp, warp_flow)        
                    

                     

                    is_print_iter = int(global_step) % print_iterations == 0              
                    is_last = epoch == epochs - 1 and i == num_frames - 2 and k == num_videos - 1
                    should_print = is_print_iter or is_last
                    
                    to_print = [epoch, global_step] + to_print 
                    
                    print 'epoch:, global_step:, style:, content:, temporal:, tv:, loss:' 
                    print to_print
                    print "training ghjhg"

                    if should_print:
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str)
                        summary_writer.flush()
                        print "saving model"
                        saver.save(sess, save_path)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


home = "/home/wangsq/CS4243/"
content_targets = home + "DAVIS/"
content_targets = _get_files(content_targets, folder = True)
style_target = home + "g.jpg"
style_target = get_img(style_target)
content_weight = CONTENT_WEIGHT
style_weight = STYLE_WEIGHT
temporal_weight = TEMPORAL_WEIGHT
             
tv_weight = TV_WEIGHT
loss_net_path = VGG_PATH
log_dir = home + "DAVIS_model/models/ghjhg_model/"
save_path= home + 'DAVIS_model/models/ghjhg_model/model.ckpt'



optimize(save_path, content_targets, style_target, content_weight, style_weight, temporal_weight, 
             tv_weight, loss_net_path, log_dir)
    