import datetime
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# ROS 노드 초기화
rospy.init_node('image_viewer', anonymous=True)


# yolo
CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

coco128 = open('./dataset/coco128.txt', 'r')
data = coco128.read()
class_list = data.split('\n')
coco128.close()

model = YOLO('./pretrained/yolov8s.pt')

# OpenCV Bridge 초기화
cv_bridge = CvBridge()

# ROS 이미지 토픽 콜백 함수
def image_callback(msg):
    global model
    
    try:

        start = datetime.datetime.now()

        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        frame = cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        detection = model(frame)[0]

        for data in detection.boxes.data.tolist(): # data : [xmin, ymin, xmax, ymax, confidence_score, class_id]
            confidence = float(data[4])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = int(data[5])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, class_list[label]+' '+str(round(confidence, 2)) + '%', (xmin, ymin), cv2.FONT_ITALIC, 1, WHITE, 2)

        end = datetime.datetime.now()

        total = (end - start).total_seconds()
        print(f'Time to process 1 frame: {total * 1000:.0f} milliseconds')

        fps = f'FPS: {1 / total:.2f}'
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # OpenCV 이미지를 화면에 표시
        cv2.imshow("ROS Image", frame)
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