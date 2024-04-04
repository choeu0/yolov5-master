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
    url_A1 = 'https://license-d68c6-default-rtdb.firebaseio.com/parking_spot_state/A1.json'
    url_A2 = 'https://license-d68c6-default-rtdb.firebaseio.com/parking_spot_state/A2.json'
    response_A1 = requests.get(url_A1)
    response_A2 = requests.get(url_A2)
    return response_A1.json(), response_A2.json()

def process_vehicle_detection(count, db, reader, save_path):
    A1_value, A2_value = get_parking_spot_state_values()
    if A1_value or A2_value:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            # 전처리 시작
            # 노이즈 제거
            frame_denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            # 대비 향상 (CLAHE 적용)
            frame_gray = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame_clahe = clahe.apply(frame_gray)
            
            # 대비 향상된 그레이스케일 이미지를 BGR로 변환하여 저장 및 사용
            frame_preprocessed = cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2BGR)
            
            img_path = os.path.join(save_path, f"captured_frame_{count}.jpg")
            cv2.imwrite(img_path, frame_preprocessed)
            print(f"전처리된 화면을 {img_path}에 캡쳐했습니다.")
            # 전처리 끝

            # 객체 탐지 실행
            command = f"python detect.py --weights best.pt --save-txt --conf 0.5 --source {img_path}"
            subprocess.run(command, shell=True)

            # 최신 실험 경로 및 레이블 처리
            latest_exp_path = get_latest_exp_path(r'C:/capstone/yolov5-master/runs/detect')
            if latest_exp_path:
                process_licenseplate_detection(latest_exp_path, frame_clahe, count, db, A1_value, A2_value, reader)
            else:
                print("객체 탐지 실험 경로를 찾을 수 없습니다.")
            cap.release()
        else:
            print("화면 캡쳐에 실패했습니다.")
        cv2.destroyAllWindows()

# 번호판 인식 함수
def process_licenseplate_detection(latest_exp_path, frame, count, db, A1_value, A2_value, reader):
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

        text_list = reader.readtext(roi, detail=0)
        combined_text = ' '.join(text_list)
        print(combined_text)  # 인식된 번호판 텍스트 출력

        # x_center 계산 및 저장
        x_center = x * frame.shape[1]
        recognized_plates.append((x_center, combined_text))

    # 주차 상태에 따른 조건 분기
    if A1_value and not A2_value:
        parking_spot = 'A1'
        update_firebase_recognition(db, A1_value, A2_value, parking_spot, combined_text)
    elif not A1_value and A2_value:
        parking_spot = 'A2'
        update_firebase_recognition(db, A1_value, A2_value, parking_spot, combined_text)
    elif A1_value and A2_value:
        # x_center 값을 기준으로 번호판 정렬
        recognized_plates.sort(key=lambda x: x[0])

        # 가장 왼쪽 번호판을 A1에, 나머지를 A2에 할당
        for i, (x_center, combined_text) in enumerate(recognized_plates):
            if i == 0:
                parking_spot = 'A1'
            else:
                parking_spot = 'A2'
            # 각 번호판에 대해 주차 상태 업데이트
            update_firebase_recognition(db, A1_value, A2_value, parking_spot, combined_text)

# 좌표 계산 함수
def calculate_coordinates(x, y, w, h, frame):
    x_min = int((x - w / 2) * frame.shape[1])
    x_max = int((x + w / 2) * frame.shape[1])
    y_min = int((y - h / 2) * frame.shape[0])
    y_max = int((y + h / 2) * frame.shape[0])
    return x_min, x_max, y_min, y_max

# Firebase 차량 등록 여부 업데이트 함수
def update_firebase_recognition(db, A1_value, A2_value, parking_spot, combined_text):
    
    if (parking_spot == 'A1' and A1_value) or (parking_spot == 'A2' and A2_value):
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
    
    while True:  # 주요 반복 루프
        process_vehicle_detection(count, db, reader, save_path)
        count += 1  # 이미지 카운터 업데이트

if __name__ == "__main__":
    main()
