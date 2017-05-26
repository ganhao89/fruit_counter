#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('fruit_counter')
import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'orange')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel')}



class fruit_counter:
  global count
  global old_gray
  global frame_gray
  global p0
  global lk_params
  global mask
  global net
  global count
  global ax
  def __init__(self):

      self.count = 0
    
      cfg.TEST.HAS_RPN = True  # Use RPN for proposals

      args = self.parse_args()

      prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
      caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

      if not os.path.isfile(caffemodel):
          raise IOError(('{:s} not found.\nDid you run ./data/script/'
                         'fetch_faster_rcnn_models.sh?').format(caffemodel))

      if args.cpu_mode:
          caffe.set_mode_cpu()
      else:
          caffe.set_mode_gpu()
          caffe.set_device(args.gpu_id)
          cfg.GPU_ID = args.gpu_id
      self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

      self.bridge = CvBridge()
      self.image_sub = rospy.Subscriber("/kinect2/qhd/image_color",Image,self.callback)
    
   
  def vis_detections(self, im, class_name, dets, thresh=0.5):
      """Draw detected bounding boxes."""
      inds = np.where(dets[:, -1] >= thresh)[0]
      if len(inds) == 0:
          return
      if self.count==0:
          self.lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
          self.frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
          self.p0 = dets[inds,:2]
          self.p0=self.p0.reshape(-1,1,2)
          print(self.p0.shape)
          self.mask = np.zeros_like(im)
      else:
          self.mask = np.zeros_like(im)
          self.frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
          p2 = dets[inds,:2]
          p2 = p2.reshape(-1,1,2)
          # calculate optical flow
          p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, self.frame_gray, self.p0, None, **self.lk_params)
          # Select good points
          good_new = p1[st==1]
          good_old = self.p0[st==1]
          # draw the tracks
          for i,(new,old) in enumerate(zip(good_new,good_old)):
              a,b = new.ravel()
              c,d = old.ravel()
              self.mask = cv2.line(self.mask, (a,b),(c,d), (0,0,255), 2)
              im = cv2.circle(im,(a,b),5,(255,0,0),-1)
          im = cv2.add(im,self.mask)
          #self.p0= good_new.reshape(-1,1,2)
          self.p0 = p2
      #for i in inds:
      #    bbox = dets[i, :4]
      #    score = dets[i, -1]
      #    cv2.rectangle(im,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
      cv2.imshow('frame',im)
      cv2.waitKey(30)
      self.count = 1
      self.old_gray = self.frame_gray.copy()

  def demo(self, net, im):
      """Detect object classes in an image using pre-computed object proposals."""
      
      # Detect all object classes and regress object bounds
      timer = Timer()
      timer.tic()
      scores, boxes = im_detect(net, im)
      timer.toc()
      print ('Detection took {:.3f}s for '
             '{:d} object proposals'.format(timer.total_time, boxes.shape[0]))

      # Visualize detections for each class
      CONF_THRESH = 0.5
      NMS_THRESH = 0.3
      for cls_ind, cls in enumerate(CLASSES[1:]):
          cls_ind += 1 # because we skipped background
          cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
          cls_scores = scores[:, cls_ind]
          dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
          keep = nms(dets, NMS_THRESH)
          dets = dets[keep, :]
          self.vis_detections(im, cls, dets, thresh=CONF_THRESH)
      

  def parse_args(self):
      """Parse input arguments."""
      parser = argparse.ArgumentParser(description='Faster R-CNN demo')
      parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
      parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
      parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

      args = parser.parse_args()

      return args


  def callback(self,data):
      caffe.set_mode_gpu()
      caffe.set_device(0)
      cfg.GPU_ID = 0

      try:
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)

      self.demo(self.net, frame)
      

if __name__ == '__main__':
    fc = fruit_counter()
    rospy.init_node('fruit_counter', anonymous=True)
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()
