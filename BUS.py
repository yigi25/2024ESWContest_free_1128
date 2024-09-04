# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base imageimport cv2

from ultralytics import YOLO
import pyttsx3
import time

# YOLOv8 모델 로드 (사전 학습된 모델)
model = YOLO("best.pt")

# pyttsx3 초기화
engine = pyttsx3.init()

# GStreamer 파이프라인 생성 함수
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            
            # 비디오 파일의 프레임 너비, 높이 및 FPS 가져오기
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)

            # 화면을 3구역으로 나누기 위한 기준점 설정
            left_bound = frame_width // 3
            right_bound = 2 * frame_width // 3

            # 결과를 저장할 비디오 파일 생성
            out = cv2.VideoWriter('output_video3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            # 마지막 음성 안내 시간을 기록할 변수 초기화
            last_announcement_time = 0
            announcement_interval = 6  # 음성 안내 반복 주기 (초 단위)

            # 문 너비 임계값 설정 (예: 프레임 너비의 10%)
            door_width_threshold = frame_width * 0.10

            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    break

                # YOLOv8 모델을 사용하여 객체 감지
                results = model(frame)

                # 감지된 객체를 프레임에 그리기
                annotated_frame = results[0].plot()

                # 현재 시간 가져오기
                current_time = time.time()

                # 특정 클래스("front door") 감지 여부 확인
                for obj in results[0].boxes:
                    cls_name = model.names[int(obj.cls)]
                    if cls_name == "front_door":
                        x1, y1, x2, y2 = [int(coord) for coord in obj.xyxy[0]]  # 텐서의 각 요소를 스칼라로 변환
                        # 객체의 중심 좌표 계산
                        center_x = (x1 + x2) // 2
                        door_width = x2 - x1

                        if center_x < left_bound:
                            position_text = "문이 왼쪽에 있습니다."
                            if door_width < door_width_threshold:
                                position_text += " 왼쪽으로 회전하여 접근하세요."
                            else:
                                position_text += " 회전할 필요가 없습니다."
                        elif center_x > right_bound:
                            position_text = "문이 오른쪽에 있습니다."
                            if door_width < door_width_threshold:
                                position_text += " 오른쪽으로 회전하여 접근하세요."
                            else:
                                position_text += " 회전할 필요가 없습니다."
                        else:
                            position_text = "문이 중앙에 있습니다."
                            if door_width < door_width_threshold:
                                position_text += " 문이 멀리 있습니다. 직진하세요."
                            else:
                                position_text += " 문이 바로 앞에 있습니다."

                        # 일정 시간이 지난 후에만 음성 안내 제공
                        if current_time - last_announcement_time >= announcement_interval:
                            print(position_text)
                            
                            # 텍스트를 음성으로 변환 및 재생
                            engine.say(position_text)
                            engine.runAndWait()

                            # 마지막 음성 안내 시간 업데이트
                            last_announcement_time = current_time

                # 결과 프레임을 화면에 표시
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, annotated_frame)
                else:
                    break

                # 결과 프레임을 비디오 파일에 저장
                out.write(annotated_frame)

                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            out.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == "__main__":
    show_camera()
