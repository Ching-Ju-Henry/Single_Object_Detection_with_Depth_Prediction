import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import matplotlib.patches as patches

import models

def predict(model_data_path, image_path):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    
    # Read image
    img = Image.open(image_path)
    size = img.size
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    
    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sess)      
        
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:               
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)
        
        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        image = pred.copy()
        #reading pred location
        f = open('/home/henry/rolo/YOLO_tensorflow/output/out_TV4.txt')
        all_list = []
        all_list = f.read()
        
        #prepare bbox information 
        new = all_list.split(',')
        x = int(new[1])
        y = int(new[2])
        w = int(new[3])
        h = int(new[4])
        
        fig = plt.figure()
        ii = plt.imshow(image[0,:,:,0], interpolation='nearest')#, interpolation='nearest'
        fig.colorbar(ii)
        currentAxis=plt.gca()
        rect=patches.Rectangle((x-w, y-h),2*w,2*h,linewidth=1,edgecolor='r',facecolor='none')
        currentAxis.add_patch(rect)
        plt.savefig('/home/henry/fcrn/FCRN-DepthPrediction/tensorflow/deep/TV4.jpg')#,dpi=90, bbox_inches='tight'

        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    
    os._exit(0)

if __name__ == '__main__':
    main()
