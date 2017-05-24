#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('fruit_counter')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class fruit_counter:
  global count
  global old_gray
  global frame_gray
  global p0
  global lk_params
  global color
  global mask
  def __init__(self):
    self.count = 0
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/kinect2/qhd/image_color",Image,self.callback)

  def callback(self,data):
    
    try:
      frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
   
    if self.count==0:
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        self.color = np.random.randint(0,255,(100,3))
        # Take first frame and find corners in it
        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
        self.p0 = cv2.goodFeaturesToTrack(self.frame_gray, mask = None, **feature_params)
        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(frame)
    
    else:
        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, self.frame_gray, self.p0, None, **self.lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(self.mask, (a,b),(c,d), self.color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,self.color[i].tolist(),-1)
        img = cv2.add(frame,self.mask)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
           print("error") 
        # Now update the previous frame and previous points
        self.p0 = good_new.reshape(-1,1,2)
     
    self.count = 1
    self.old_gray = self.frame_gray.copy()




def main(args):
  
  fc = fruit_counter()
  rospy.init_node('fruit_counter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
