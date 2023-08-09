import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# ROS 노드 초기화
rospy.init_node('image_viewer', anonymous=True)

# OpenCV Bridge 초기화
cv_bridge = CvBridge()

# ROS 이미지 토픽 콜백 함수
def image_callback(msg):
    try:
        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        # OpenCV 이미지를 화면에 표시
        cv2.imshow("ROS Image", cv_image)
        cv2.waitKey(1)  # 키 입력 대기 (1ms)

    except Exception as e:
        print(e)

# ROS 이미지 토픽 구독
image_topic = "/iris/camera/rgb/image_raw"  # 실제 토픽명에 맞게 수정
rospy.Subscriber(image_topic, Image, image_callback)

# 프로그램 종료 시 OpenCV 창 닫기
rospy.on_shutdown(cv2.destroyAllWindows)

# ROS 루프 실행
rospy.spin()