import cv2
import os
import subprocess
import glob
import easyocr
import firebase_admin
from firebase_admin import credentials, firestore, db
import tkinter as tk
import requests
import numpy as np

# Firebase 프로젝트 설정 함수
def initialize_firebase():
    cred = credentials.Certificate("license-d68c6-firebase-adminsdk-rvher-328ef35868.json")
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://license-d68c6-default-rtdb.firebaseio.com'})
    return firestore.client()

# 메시지 창 표시 함수
def show_message(result):
    popup = tk.Tk()
    popup.title("차량 확인 결과")
    label = tk.Label(popup, text=result)
    label.pack(padx=10, pady=10)
    close_button = tk.Button(popup, text="닫기", command=popup.destroy)
    close_button.pack(pady=10)
    popup.mainloop()

# 최신 실험 경로 가져오기 함수
def get_latest_exp_path(detect_path):
    exp_folders = glob.glob(os.path.join(detect_path, 'exp*'))
    return max(exp_folders, key=os.path.getctime) if exp_folders else None

# 주차 상태 값 가져오기 함수
def get_parking_spot_state_values():
    ref = db.reference('parking_spot_state')
    parking_spot_state = ref.get()
    # 각 주차 구역의 상태를 불러옵니다.
    A1_value = parking_spot_state.get('A1', False)
    A2_value = parking_spot_state.get('A2', False)
    A3_value = parking_spot_state.get('A3', False)
    A4_value = parking_spot_state.get('A4', False)
    return A1_value, A2_value, A3_value, A4_value

def get_parking_spot_angle():
    ref = db.reference('parking_spot_angle/current_angle')
    current_angle = ref.get()
    return current_angle

def set_webcam_active_status(active):
    # Firebase 데이터베이스에 웹캠의 활성화 상태 설정
    ref = firebase_admin.db.reference('webcam_active')
    ref.set(active)

def process_vehicle_detection(count, db, reader, save_path, cap):
    if not hasattr(process_vehicle_detection, "previous_angle"):
        process_vehicle_detection.previous_angle = None  # 함수에 상태 저장을 위한 속성 초기화

    current_angle = get_parking_spot_angle()
    
    if current_angle != process_vehicle_detection.previous_angle:
        process_vehicle_detection.previous_angle = current_angle  # 각도 상태 업데이트
        # cap = cv2.VideoCapture(0)
        
        set_webcam_active_status(True)  # 웹캠 활성화 상태를 True로 설정
        
        # 줌 수준 설정 (0.0부터 1.0까지의 범위, 1.0은 최대 줌)
        # zoom_level = 1.0  # 예시로 0.5로 설정
        # cap.set(cv2.CAP_PROP_ZOOM, zoom_level)  # 줌 레벨 설정
        
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 너비를 1920으로 설정
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 높이를 1080으로 설정
        
        ret, frame = cap.read()
        if ret:
            # 전처리 시작
            # # 리사이징 파라미터
            # scale_factor = 2  # 이미지 크기를 1.5배 증가시킵니다.
            # width = int(frame.shape[1] * scale_factor)
            # height = int(frame.shape[0] * scale_factor)

            # # 이미지 리사이징
            # resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # 노이즈 제거
            frame_denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            # 대비 향상 (CLAHE 적용)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame_clahe = clahe.apply(frame_gray)
            
            # 대비 향상된 그레이스케일 이미지를 BGR로 변환하여 저장 및 사용
            frame_preprocessed = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            
            img_path = os.path.join(save_path, f"captured_frame_{count}.jpg")
            cv2.imwrite(img_path, frame_preprocessed)
            print(f"전처리된 화면을 {img_path}에 캡쳐했습니다.")
            # 전처리 끝

            # 객체 탐지 실행
            command = f"python detect.py --weights best_2.pt --save-txt --conf 0.5 --source {img_path}"
            subprocess.run(command, shell=True)

            # 최신 실험 경로 및 레이블 처리
            latest_exp_path = get_latest_exp_path(r'C:/capstone/yolov5-master/runs/detect')
            if latest_exp_path:
                process_licenseplate_detection(latest_exp_path, frame_clahe, count, db, reader, save_path, current_angle)
            else:
                print("객체 탐지 실험 경로를 찾을 수 없습니다.")
        else:
            print("화면 캡쳐에 실패했습니다.")

        set_webcam_active_status(False) # 웹캠 활성화 상태를 False로 설정

# 번호판 인식 함수
def process_licenseplate_detection(latest_exp_path, frame, count, db, reader, save_path, current_angle):
    A1_value, A2_value, A3_value, A4_value = get_parking_spot_state_values()
    parking_values = {'A1': A1_value, 'A2': A2_value, 'A3': A3_value, 'A4': A4_value}
    
    labels_path = os.path.join(latest_exp_path, 'labels', f"captured_frame_{count}.txt")
    with open(labels_path, 'r') as file:
        lines = file.readlines()

    # 인식된 번호판 정보를 저장할 리스트
    recognized_plates = []

    for i, line in enumerate(lines):
        parts = line.strip().split()
        x, y, w, h = map(float, parts[1:5])
        x_min, x_max, y_min, y_max = calculate_coordinates(x, y, w, h, frame)
        roi = frame[y_min:y_max, x_min:x_max]
        
        plate_img_path = os.path.join(save_path, f"plate_{count}_{i}.jpg")
        cv2.imwrite(plate_img_path, roi)
        print(f"번호판 이미지를 {plate_img_path}에 저장했습니다.")
        
        text_list = reader.readtext(roi, detail=0)
        combined_text = ' '.join(text_list)
        combined_text = combined_text[:9]
        print(combined_text)  # 인식된 번호판 텍스트 출력

        # x_center 계산 및 저장
        x_center = x * frame.shape[1]
        recognized_plates.append((x_center, combined_text))

    # 주차 상태에 따른 조건 분기
    
    if current_angle == 0:
    # 주차 상태에 따른 조건 분기
        if A1_value and not A2_value:
            parking_spot = 'A1'
            update_firebase_recognition(db, parking_values, 'A1', combined_text)
        elif not A1_value and A2_value:
            parking_spot = 'A2'
            update_firebase_recognition(db, parking_values, 'A2', combined_text)
        elif A1_value and A2_value:
            # x_center 값을 기준으로 번호판 정렬
            recognized_plates.sort(key=lambda x: x[0])

            # 가장 왼쪽 번호판을 A2에, 나머지를 A1에 할당
            for i, (x_center, combined_text) in enumerate(recognized_plates):
                if i == 0:
                    parking_spot = 'A2'
                else:
                    parking_spot = 'A1'
                # 각 번호판에 대해 주차 상태 업데이트
                update_firebase_recognition(db, parking_values, parking_spot, combined_text)

    elif current_angle == 180:
        # 주차 상태에 따른 조건 분기
        if A3_value and not A4_value:
            parking_spot = 'A3'
            update_firebase_recognition(db, parking_values, 'A3', combined_text)
        elif not A3_value and A4_value:
            parking_spot = 'A4'
            update_firebase_recognition(db, parking_values, 'A4', combined_text)
        elif A3_value and A4_value:
            # x_center 값을 기준으로 번호판 정렬
            recognized_plates.sort(key=lambda x: x[0])

            # 가장 왼쪽 번호판을 A3에, 나머지를 A4에 할당
            for i, (x_center, combined_text) in enumerate(recognized_plates):
                if i == 0:
                    parking_spot = 'A3'
                else:
                    parking_spot = 'A4'
                # 각 번호판에 대해 주차 상태 업데이트
                update_firebase_recognition(db, parking_values, parking_spot, combined_text)

# 좌표 계산 함수
def calculate_coordinates(x, y, w, h, frame):
    x_min = int((x - w / 2) * frame.shape[1])
    x_max = int((x + w / 2) * frame.shape[1])
    y_min = int((y - h / 2) * frame.shape[0])
    y_max = int((y + h / 2) * frame.shape[0])
    return x_min, x_max, y_min, y_max

def update_firebase_recognition(db, parking_values, parking_spot, combined_text):
    # parking_values는 { 'A1': A1_value, 'A2': A2_value, 'A3': A3_value, 'A4': A4_value } 형태의 딕셔너리입니다.
    spot_active = parking_values.get(parking_spot, False)
    
    if spot_active:
        registered_ref = db.collection("registered").document("car_license")
        registered_doc = registered_ref.get()
        if registered_doc.exists:
            registered_data = registered_doc.to_dict()
            is_registered = combined_text in registered_data.values()
            ref = firebase_admin.db.reference(f'/parking_spot_registered/{parking_spot}')
            ref.set(is_registered)  # 차량 등록 여부 업데이트
            status = "등록차량" if is_registered else "비등록차량"
            show_message(f"{parking_spot}에 {status}입니다.")
        else:
            ref = firebase_admin.db.reference(f'/parking_spot_registered/{parking_spot}')
            ref.set(False)  # 비등록차량으로 설정
            show_message(f"{parking_spot}에 비등록차량입니다.")
    else:
        print(f"{parking_spot} 구역은 비어 있거나 번호판이 정확히 인식되지 않았습니다.")

def main():
    db = initialize_firebase()  # Firebase 초기화 및 인스턴스 획득
    count = 0  # 이미지 카운터 초기화
    reader = easyocr.Reader(['ko'], gpu=False)  # OCR 초기화
    save_path = r'C:/capstone/yolov5-master/assets'  # 저장 경로 설정
    
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    try:
        while True:
            process_vehicle_detection(count, db, reader, save_path, cap)
            count += 1
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.
        
if __name__ == "__main__":
    main()
