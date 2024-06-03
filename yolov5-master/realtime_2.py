import cv2
import os
import subprocess
import glob
import easyocr
import firebase_admin
from firebase_admin import credentials, firestore, db
import tkinter as tk
import threading
import time

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

# 가장 최근의 YOLOv5 탐지 경로를 가져오는 함수
def get_latest_exp_path(detect_path):
    exp_folders = glob.glob(os.path.join(detect_path, 'exp*'))
    if exp_folders:
        latest_exp_path = max(exp_folders, key=os.path.getctime)
        return latest_exp_path
    else:
        return None

# 주차 상태 값 가져오기 함수
def get_parking_spot_state_values():
    ref = db.reference('parking_spot_state')
    parking_spot_state = ref.get()

    A1_value = parking_spot_state.get('A1', False)
    A2_value = parking_spot_state.get('A2', False)
    A3_value = parking_spot_state.get('A3', False)
    A4_value = parking_spot_state.get('A4', False)
    return A1_value, A2_value, A3_value, A4_value

# 주차 구역의 현재 각도 가져오는 함수
def get_parking_spot_angle():
    ref = db.reference('parking_spot_angle/current_angle')
    current_angle = ref.get()
    return current_angle

# 인식된 차량 번호판 정보를 Firebase에 업데이트하는 함수
def update_firebase_recognition(firestore_db, combined_text, count_suffix):
    # Firebase에서 등록된 차량 번호판 정보 가져오기
    registered_ref = firestore_db.collection("registered").document("car_license")
    registered_doc = registered_ref.get()
    
    if registered_doc.exists:
        registered_data = registered_doc.to_dict()
        
        # 인식된 번호판이 등록된 차량인지 확인
        for field, value in registered_data.items():
            if value == combined_text:
                ref = db.reference(f'/parking_spot_registered/{count_suffix}')
                ref.set(True)
                status = "등록차량"
                show_message(f"{count_suffix}에 {status}입니다.")
                update_parking_spot_info(count_suffix, combined_text)
                return True
        # 등록되지 않은 차량 처리
        ref = db.reference(f'/parking_spot_registered/{count_suffix}')
        ref.set(False)
        status = "비등록차량"
        show_message(f"{count_suffix}에 {status}입니다.")
        update_parking_spot_info(count_suffix, combined_text)
        return False
    else:
        # 등록된 차량 정보가 없는 경우 처리
        ref = db.reference(f'/parking_spot_registered/{count_suffix}')
        ref.set(False)
        status = "비등록차량"
        show_message(f"{count_suffix}에 {status}입니다.")
        update_parking_spot_info(count_suffix, combined_text)
        return False

# 주차 구역 정보 업데이트 함수
def update_parking_spot_info(count_suffix, combined_text):
    ref = firebase_admin.db.reference('parking_spot_info')
    ref.update({count_suffix: combined_text})

# 비동기 프레임 처리 함수
def process_frame_async(frame, count, firestore_db, current_angle, parking_values):
    threading.Thread(target=process_frame, args=(frame, count, firestore_db, current_angle, parking_values)).start()

# 프레임 처리 함수
def process_frame(frame, count, firestore_db, current_angle, parking_values):
    save_path = r'C:/capstone/yolov5-master/assets'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 현재 프레임 이미지로 저장
    img_path = os.path.join(save_path, f"captured_frame_{count}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"화면을 {img_path}에 캡쳐했습니다.")

    # YOLOv5 탐지를 위한 명령어 실행
    command = f"python detect.py --weights best_2.pt --save-txt --conf 0.5 --source {img_path} --device cpu"
    subprocess.run(command, shell=True)

    # YOLOv5 탐지 결과 경로 가져오기
    latest_exp_path = get_latest_exp_path(r'C:\capstone\yolov5-master\runs\detect')
    if latest_exp_path:
        print(f"Latest experiment path: {latest_exp_path}")
        file_path = os.path.join(latest_exp_path, 'labels', f"captured_frame_{count}.txt")
        if os.path.exists(file_path):
            print(f"Label file found: {file_path}")
        else:
            print(f"Label file not found: {file_path}")
            return
    else:
        print("No experiment path found, skipping this frame.")
        return

    x_list, y_list, w_list, h_list = [], [], [], []
    
    # YOLOv5 탐지 결과 파일에서 번호판 위치 정보 읽기
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # 인식된 번호판 정보를 저장할 리스트
        recognized_plates = []
        for line in lines:
            data = line.strip().split(' ')
            x = float(data[1])
            y = float(data[2])
            w = float(data[3])
            h = float(data[4])
            x_list.append(x)
            y_list.append(y)
            w_list.append(w)
            h_list.append(h)

    for i in range(len(x_list)):
        x_min = int((x_list[i] - w_list[i] / 2) * frame.shape[1])
        y_min = int((y_list[i] - h_list[i] / 2) * frame.shape[0])
        x_max = int((x_list[i] + w_list[i] / 2) * frame.shape[1])
        y_max = int((y_list[i] + h_list[i] / 2) * frame.shape[0])
        roi = frame[y_min:y_max, x_min:x_max]
        new_img_path = os.path.join(save_path, f"cropped_frame_{count}_{i}.jpg")
        cv2.imwrite(new_img_path, roi)
        print(f"ROI를 {new_img_path}에 저장했습니다.")

        # EasyOCR을 사용하여 번호판 텍스트 인식
        reader = easyocr.Reader(['ko'], gpu=True)
        img = cv2.imread(new_img_path)
        text_list = reader.readtext(img, detail=0)
        combined_text = ' '.join(text_list)[:9]
        print(combined_text)
        
        # x_center 계산 및 저장
        x_center = x_list[i] * frame.shape[1]
        recognized_plates.append((x_center, combined_text))

    # 주차 상태에 따른 조건 분기 및 인식 결과 업데이트
    if current_angle == 0:
        if parking_values[0] and not parking_values[1]:
            parking_spot = 'A1'
            update_firebase_recognition(firestore_db, recognized_plates[0][1], parking_spot)

        elif not parking_values[0] and parking_values[1]:
            parking_spot = 'A2'
            update_firebase_recognition(firestore_db, recognized_plates[0][1], parking_spot)

        elif parking_values[0] and parking_values[1]:
            # x_center 값을 기준으로 번호판 정렬
            recognized_plates.sort(key=lambda x: x[0])
            
            # 가장 왼쪽 번호판을 A2에, 나머지를 A1에 할당
            for i, (x_center, combined_text) in enumerate(recognized_plates):
                parking_spot = 'A1' if i == 0 else 'A2'
                update_firebase_recognition(firestore_db, combined_text, parking_spot)

    elif current_angle == 180:
        if parking_values[2] and not parking_values[3]:
            parking_spot = 'A3'
            update_firebase_recognition(firestore_db, recognized_plates[0][1], parking_spot)

        elif not parking_values[2] and parking_values[3]:
            parking_spot = 'A4'
            update_firebase_recognition(firestore_db, recognized_plates[0][1], parking_spot)

        elif parking_values[2] and parking_values[3]:
            recognized_plates.sort(key=lambda x: x[0])
            
            for i, (x_center, combined_text) in enumerate(recognized_plates):
                parking_spot = 'A3' if i == 0 else 'A4'
                update_firebase_recognition(firestore_db, combined_text, parking_spot)
    

def main():
    firestore_db = initialize_firebase()
    external_camera_index = 0
    cap = cv2.VideoCapture(external_camera_index)
    count = 0

    # 각 주차 구역별로 처리 상태 저장
    processing = [False, False, False, False]

    while True:
        A1_value, A2_value, A3_value, A4_value = get_parking_spot_state_values()
        parking_values = [A1_value, A2_value, A3_value, A4_value]
        current_angle = get_parking_spot_angle()
        
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 각도에 따라 특정 주차 구역만 고려
        if current_angle == 0:
            for i in [0, 1]:  # A1과 A2만 고려
                if parking_values[i] and not processing[i]:
                    processing[i] = True
                    time.sleep(2)   # 회전을 고려하여 2초 대기
                    process_frame_async(frame, count, firestore_db, current_angle, parking_values)
                    count += 1
                elif not parking_values[i]:
                    processing[i] = False
        elif current_angle == 180:
            for i in [2, 3]:  # A3와 A4만 고려
                if parking_values[i] and not processing[i]:
                    processing[i] = True
                    time.sleep(2)
                    process_frame_async(frame, count, firestore_db, current_angle, parking_values)
                    count += 1
                elif not parking_values[i]:
                    processing[i] = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

