#include <Wire.h>
#include <Firebase_Arduino_WiFiNINA.h>
#include <NewPing.h>
#include <Servo.h>

// Firebase 및 WiFi 설정 정보
#define FIREBASE_HOST "license-d68c6-default-rtdb.firebaseio.com"
#define FIREBASE_AUTH "qyQJdDEJv7Kn61As6rjle22QmTzViiHIGoGLC0ba"
#define SSID "iptime0"
#define SSID_PASSWORD "dndud516"
// #define SSID "uyeong"
// #define SSID_PASSWORD "dndud516"
#define PARKING_SPOT_STATE_KEY "/parking_spot_state"
#define PARKING_SPOT_ANGLE_KEY "/parking_spot_angle/current_angle"

// 센서 및 서보 모터 설정
#define NUM_SENSORS 4
const int trigPins[NUM_SENSORS] = {2, 4, 6, 8};
const int echoPins[NUM_SENSORS] = {3, 5, 7, 9};
const int SERVO_PIN = 10;
const unsigned int MAX_DISTANCE = 200;
const int STANDARD_DISTANCE = 15;

Servo myservo;
FirebaseData firebaseData;
NewPing sonar[NUM_SENSORS] = { // 초음파 센서 객체 배열
    NewPing(trigPins[0], echoPins[0], MAX_DISTANCE),
    NewPing(trigPins[1], echoPins[1], MAX_DISTANCE),
    NewPing(trigPins[2], echoPins[2], MAX_DISTANCE),
    NewPing(trigPins[3], echoPins[3], MAX_DISTANCE)
};

// 각 센서의 현재 상태
bool currentStates[NUM_SENSORS] = {false, false, false, false}; 

void setup() {
    Serial.begin(9600); // 시리얼 통신 시작
    myservo.attach(SERVO_PIN); // 서보 모터 핀 연결
    WiFi.begin(SSID, SSID_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi");
    Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH, SSID, SSID_PASSWORD);
}

void updateParkingSpotState(int sensorIndex, bool isActive) {
    String path = String(PARKING_SPOT_STATE_KEY) + "/A" + String(sensorIndex + 1); // Firebase 경로 설정
    Firebase.setBool(firebaseData, path, isActive); // 상태 업데이트
}

void updateServoAngle(int angle) { // Firebase에 각도 업데이트
    Firebase.setInt(firebaseData, PARKING_SPOT_ANGLE_KEY, angle);
}

void loop() {
    int lastActiveSensor = -1; // 마지막으로 활성화된 센서 인덱스 저장

    for (int i = 0; i < NUM_SENSORS; i++) {
        unsigned int distance = sonar[i].ping_cm(); // 거리 측정
        bool isActive = distance > 0 && distance <= STANDARD_DISTANCE; // 센서 활성화 여부 결정

        if (currentStates[i] != isActive) { // 상태 변경이 있을 때만 업데이트
            currentStates[i] = isActive;
            updateParkingSpotState(i, isActive); // Firebase에 상태 업데이트
            Serial.print("Sensor ");
            Serial.print(i + 1);
            Serial.print(": ");
            Serial.print(distance);
            Serial.println(" cm");

            if (isActive) { // 센서가 활성화된 경우
                lastActiveSensor = i; // 마지막 활성화된 센서로 기록
            }
        }
    }

    if (lastActiveSensor != -1) { // 활성화된 센서가 있는 경우
        int angle = lastActiveSensor < 2 ? 0 : 180; // A1 또는 A2는 0도, A3 또는 A4는 180도
        myservo.write(angle);  // 서보 모터 각도 조정
        Serial.print("Rotating to ");
        Serial.print(angle);
        Serial.println(" degrees");
        updateServoAngle(angle); // Firebase에 각도 업데이트
    }

    delay(1000); // 다음 루프까지 1초 대기
}
