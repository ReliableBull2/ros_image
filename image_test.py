#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from utils import ARUCO_DICT, aruco_display
# gimbal_control 
import asyncio
from mavsdk import System
from mavsdk.gimbal import GimbalMode, ControlMode



# yaw
errorY_last = 0
outputY =0

# pitch
errorP_last = 0
outputP =0

  # Print "Hello!" to terminal

  # Initialize the ROS Node named 'opencv_example', allow multiple nodes to be run with this name
rospy.init_node('opencv_example', anonymous=True)

  # Print "Hello ROS!" to the Terminal and to a ROS Log file located in ~/.ros/log/loghash/*.log
rospy.loginfo("Hello ROS!")

  # Initialize the CvBridge class
bridge = CvBridge()

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_ARUCO_ORIGINAL"])
arucoParams = cv2.aruco.DetectorParameters_create()

  # Define a function to show the image in an OpenCV Window
def show_image(img):
    cv2.imshow("Image Window", img)
    
    cv2.waitKey(1)

  # Define a callback for the Image message
def image_callback(img_msg):

    global errorY_last   
    global outputY
    global errorP_last   
    global outputP

    # log some info about the image topic
    #rospy.loginfo(img_msg.header)

      # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8") # color
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Flip the image 90deg
   # cv_image = cv2.transpose(cv_image)
   # cv_image = cv2.flip(cv_image,1)



    #convert to HSV
    #hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
    #lower_red = np.array([150, 50, 50])        # 빨강색 범위
    #upper_red = np.array([180, 255, 255])

    #mask = cv2.inRange(hsv, lower_red, upper_red)

    #print(np.transpose(mask.nonzero()))

    #points = np.transpose(mask.nonzero())
    
    #cx = int((points[:,0].min()+points[:,0].max())/2)
    #cy = int((points[:,1].min() + points[:,1].max())/2)

   # print("x: " , cx , " y : " , cy)


    #res = cv2.bitwise_and(cv_image, cv_image, mask=mask)

    #centerPoint = cv_image.shape
    #print(centerPoint)
    #print('img.shape ', cv_image.shape)

    # draw center point
    #cv_image = cv2.circle(cv_image , (cy,cx) , radius=0 , color=(255,0,0) , thickness=3)
    #cv_image = cv2.circle(cv_image , (int(centerPoint[1]/2),int(centerPoint[0]/2)) , radius=0 , color=(0,0,0) , thickness=3)

    #cv_image = cv2.rectangle(cv_image , (points[:,1].min() , points[:,0].min()),(points[:,1].max() , points[:,0].max()) ,color=(0,0,0) , thickness=2 )
      # Show the converted image

    h, w, _ = cv_image.shape

    #width=1000
    #height = int(width*(h/w))
    #cv_image = cv2.resize(cv_image, (width, height), interpolation=cv2.INTER_CUBIC)
    corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, arucoDict, parameters=arucoParams)

    (m_x , m_y , detected_markers) = aruco_display(corners, ids, rejected, cv_image)

    #print(h , w)
    #print("-------")
    #print(m_y , m_x)

  
    # m_y , m_x -> marker center y , x
    # yaw PID
    # if aruco marker detected

    maxVal = 20
    if m_x != -1:

      # yaw
      error = int(w/2) - m_x
      d_error = (error - errorY_last)
      errorY_last = d_error  

      KP_y = 2.0
      KI_y = 0.0
      KD_y = 0.0000002

      if int(error) == 0:
        outputY = 0
      elif error > 0: # right marker
        outputY = KP_y * error + KD_y * d_error

        if outputY >= maxVal:
          outputY = maxVal
        outputY = -outputY
        
      else:
        outputY = KP_y * error + KD_y * d_error

        if outputY <= -maxVal:
          outputY = -maxVal
        outputY = -outputY


      # pitch
      errorP = int(h/2) - m_y
      p_error = (errorP - errorP_last)
      errorP_last = p_error  

      KP_p = 2.0
      KI_p = 0.0
      KD_p = 0.0000002

      if int(errorP) == 0:
        outputP = 0
      elif errorP > 0: # right marker
        outputP = KP_p * errorP + KD_p * p_error

        if outputP >= maxVal:
          outputP = maxVal
        outputP = outputP
        
      else:
        outputP = KP_p * errorP + KD_p * p_error

        if outputP <= -maxVal:
          outputP = -maxVal
        outputP = outputP


      print(outputP)
          # pitch
      #errorP_last = 0
      #outputP =0


      #print(error)

    
    cv2.circle(detected_markers, (int(w/2), int(h/2)), 2, (255, 0, 0), -1)
    #drone.gimbal.set_pitch_rate_and_yaw_rate(0, -10)
    show_image( detected_markers)

# start gimbal control method


a = 0.0
async def run():
    # Init the drone

    global outputY
    global outputP
    global a
    drone = System()
    await drone.connect(system_address="udp://:14570")

    # Start printing gimbal position updates
    print_gimbal_position_task = \
        asyncio.ensure_future(print_gimbal_position(drone))

    print("Taking control of gimbal")
    await drone.gimbal.take_control(ControlMode.PRIMARY)

    #print(outputY)

   

    
    #await drone.gimbal.set_pitch_and_yaw(-90, 0)
    while 1:
        #a = a+1
        #await drone.gimbal.set_pitch_rate_and_yaw_rate(outputP , outputY)
        await asyncio.sleep(0.1)
        True

async def print_gimbal_position(drone):
    # Report gimbal position updates asynchronously
    # Note that we are getting gimbal position updates in
    # euler angles; we can also get them as quaternions
    async for angle in drone.telemetry.camera_attitude_euler():
      print()
      #  print(f"Gimbal pitch: {angle.pitch_deg}, yaw: {angle.yaw_deg}")

# end gimbal control method

  # Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
#sub_image = rospy.Subscriber("/standard_vtol/camera/rgb/image_raw", Image, image_callback)
sub_image = rospy.Subscriber("/cgo3_camera/image_raw", Image, image_callback)

  # Initialize an OpenCV Window named "Image Window"
cv2.namedWindow("Image Window", 1)

  # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed

asyncio.run(run())
while not rospy.is_shutdown():
    rospy.spin()