import cv2
import mediapipe as mp
import numpy as np
import math
import os
from PIL import ImageFont, ImageDraw, Image

# --- MediaPipe 초기화 ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh # 얼굴 인식을 위한 Face Mesh 추가

# --- 안경 이미지 로드 ---
glasses_path = "./image.png"

if not os.path.exists(glasses_path):
    print(f"오류: 안경 이미지를 찾을 수 없습니다. 경로를 확인해주세요: {glasses_path}")
    glasses_orig = np.ones((100, 300, 4), dtype=np.uint8) * 255
    cv2.putText(glasses_orig, "Image Not Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
else:
    glasses_orig = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

if glasses_orig is None:
    print(f"오류: 안경 이미지를 불러올 수 없습니다: {glasses_path}")
    exit()
else:
    # 안경 이미지가 RGBA 채널을 가지고 있는지 확인하고, 없으면 추가합니다.
    if glasses_orig.shape[2] == 3:
        glasses_orig = cv2.cvtColor(glasses_orig, cv2.COLOR_BGR2BGRA)
    glasses_orig = cv2.flip(glasses_orig, 0) # 안경 상하 반전
    glasses_orig = cv2.flip(glasses_orig, 1) # 안경 좌우 반전

# --- 일본어 텍스트 렌더링을 위한 폰트 설정 ---
font_path = "./Paperlogy-9Black.ttf"
if not os.path.exists(font_path):
    print(f"경고: 일본어 폰트를 찾을 수 없습니다. ({font_path}) 텍스트가 표시되지 않을 수 있습니다.")
    font_path = None # 폰트가 없으면 None으로 설정

# 웹캠 입력
cap = cv2.VideoCapture(0)

def get_hand_angle(hand_landmarks):
    """손의 각도를 계산하는 함수"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    angle_rad = math.atan2(middle_finger_mcp.y - wrist.y, middle_finger_mcp.x - wrist.x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# --- 모델 로드 ---
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands, \
    mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        h, w, _ = image.shape
        image = cv2.flip(image, 1)
        
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        hand_results = hands.process(image_rgb)
        face_results = face_mesh.process(image_rgb)

        image.flags.writeable = True

        breath_active = False
        
        # --- 얼굴 랜드마크 및 안경 합성 ---
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # (안경 합성 로직)
                left_iris_indices = list(range(473, 478))
                right_iris_indices = list(range(468, 473))

                left_iris_landmarks = [face_landmarks.landmark[i] for i in left_iris_indices]
                left_iris_x = int(np.mean([p.x for p in left_iris_landmarks]) * w)
                left_iris_y = int(np.mean([p.y for p in left_iris_landmarks]) * h)

                right_iris_landmarks = [face_landmarks.landmark[i] for i in right_iris_indices]
                right_iris_x = int(np.mean([p.x for p in right_iris_landmarks]) * w)
                right_iris_y = int(np.mean([p.y for p in right_iris_landmarks]) * h)

                glasses_center_x = (left_iris_x + right_iris_x) // 2
                glasses_center_y = (left_iris_y + right_iris_y) // 2
                glasses_width = int(math.dist((left_iris_x, left_iris_y), (right_iris_x, right_iris_y)) * 2.2)

                delta_x = right_iris_x - left_iris_x
                delta_y = right_iris_y - left_iris_y
                angle_rad = math.atan2(delta_y, delta_x)
                angle_deg = -math.degrees(angle_rad)

                h_g, w_g, _ = glasses_orig.shape
                
                if glasses_width > 0:
                    aspect_ratio = h_g / w_g
                    glasses_height = int(glasses_width * aspect_ratio)
                    resized_glasses = cv2.resize(glasses_orig, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
                else:
                    resized_glasses = glasses_orig.copy()

                (h_r, w_r) = resized_glasses.shape[:2]
                center_g = (w_r // 2, h_r // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center_g, angle_deg, 1.0)

                cos = np.abs(rotation_matrix[0, 0])
                sin = np.abs(rotation_matrix[0, 1])

                new_w = int((h_r * sin) + (w_r * cos))
                new_h = int((h_r * cos) + (w_r * sin))

                rotation_matrix[0, 2] += (new_w / 2) - center_g[0]
                rotation_matrix[1, 2] += (new_h / 2) - center_g[1]

                rotated_glasses = cv2.warpAffine(resized_glasses, rotation_matrix, (new_w, new_h))

                (h_rot, w_rot) = rotated_glasses.shape[:2]
                x1 = glasses_center_x - (w_rot // 2)
                y1 = glasses_center_y - (h_rot // 2)
                
                x1_b = max(x1, 0)
                y1_b = max(y1, 0)
                x2_b = min(x1 + w_rot, w)
                y2_b = min(y1 + h_rot, h)

                img_roi = image[y1_b:y2_b, x1_b:x2_b]
                
                x1_g = max(0, -x1)
                y1_g = max(0, -y1)
                x2_g = x1_g + (x2_b - x1_b)
                y2_g = y1_g + (y2_b - y1_b)
                
                glasses_roi = rotated_glasses[y1_g:y2_g, x1_g:x2_g]

                if img_roi.shape[0] > 0 and img_roi.shape[1] > 0 and glasses_roi.shape[0] > 0 and glasses_roi.shape[1] > 0:
                    if glasses_roi.shape[2] == 4:
                        alpha_channel = glasses_roi[:, :, 3] / 255.0
                        alpha_channel_inv = 1.0 - alpha_channel
                        
                        if img_roi.shape[:2] == glasses_roi.shape[:2]:
                            for c in range(3):
                                img_roi[:, :, c] = (alpha_channel * glasses_roi[:, :, c] +
                                                    alpha_channel_inv * img_roi[:, :, c])

                # --- 입 벌림 감지 및 브레스 효과 ---
                lip_top = face_landmarks.landmark[13]
                lip_bottom = face_landmarks.landmark[14]
                
                mouth_opening_distance = abs(lip_top.y - lip_bottom.y)
                
                mouth_open_threshold = 0.04 

                if mouth_opening_distance > mouth_open_threshold:
                    breath_active = True
                    mouth_center_x = int(((face_landmarks.landmark[61].x + face_landmarks.landmark[291].x) / 2) * w)
                    mouth_center_y = int(((lip_top.y + lip_bottom.y) / 2) * h)
                    
                    breath_radius = 60
                    overlay = image.copy()
                    cv2.circle(overlay, (mouth_center_x, mouth_center_y), breath_radius, (0, 0, 255), -1)
                    
                    alpha = 0.5
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                    cv2.circle(image, (mouth_center_x, mouth_center_y), breath_radius, (0, 0, 200), 5)
                else:
                    breath_active = False

        # --- 손 추적 기능 ---
        if hand_results.multi_hand_landmarks:
            
            # --- 빔 발사 자세 감지 로직 ---
            if len(hand_results.multi_hand_landmarks) == 2:
                left_hand_landmarks, right_hand_landmarks = None, None
                
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    if handedness.classification[0].label == "Left": left_hand_landmarks = hand_landmarks
                    elif handedness.classification[0].label == "Right": right_hand_landmarks = hand_landmarks

                if left_hand_landmarks and right_hand_landmarks:
                    left_thumb_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    left_index_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    right_thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    right_index_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    dist_index = math.hypot(left_index_tip.x - right_index_tip.x, left_index_tip.y - right_index_tip.y)
                    dist_thumb = math.hypot(left_thumb_tip.x - right_thumb_tip.x, left_thumb_tip.y - right_thumb_tip.y)

                    threshold = 0.1
                    if (dist_index < threshold and dist_thumb < threshold) or breath_active:
                        
                        # --- 화면 흔들림 효과 ---
                        shake_intensity = 50
                        shake_x = np.random.randint(-shake_intensity, shake_intensity + 1)
                        shake_y = np.random.randint(-shake_intensity, shake_intensity + 1)
                        
                        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
                        
                        image = cv2.warpAffine(image, M, (w, h))
                        # --- 화면 흔들림 효과 끝 ---

                        center_x = int(((left_index_tip.x + right_index_tip.x + left_thumb_tip.x + right_thumb_tip.x) / 4) * w)
                        center_y = int(((left_index_tip.y + right_index_tip.y + left_thumb_tip.y + right_thumb_tip.y) / 4) * h)
                        
                        radius = 80
                        
                        overlay = image.copy()
                        cv2.circle(overlay, (center_x, center_y), radius, (0, 255, 255) if not breath_active else (0, 0, 255), -1)
                        
                        alpha = 0.4 
                        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                        
                        cv2.circle(image, (center_x, center_y), radius, (0, 200, 200) if not breath_active else (0, 0, 200), 5)

                        # --- 손 각도에 맞춰 텍스트 회전 및 표시 ---
                        if font_path:
                            try:
                                # 두 손의 각도를 계산하고 평균을 냅니다.
                                left_angle = get_hand_angle(left_hand_landmarks)
                                right_angle = get_hand_angle(right_hand_landmarks)
                                avg_angle = (left_angle + right_angle) / 2
                                
                                beam_text = "ビーム----!" if not breath_active else "@%!&#*%!@*#&!@)(#*)"
                                font_size = 50
                                font = ImageFont.truetype(font_path, font_size)

                                # 텍스트를 그릴 별도의 투명 이미지 생성
                                bbox = font.getbbox(beam_text)
                                text_w = bbox[2] - bbox[0]
                                text_h = bbox[3] - bbox[1]
                                
                                text_img = Image.new('RGBA', (text_w, text_h), (255, 255, 255, 0))
                                text_draw = ImageDraw.Draw(text_img)
                                text_draw.text((0, 0), beam_text, font=font, fill=(255, 0, 255, 255))

                                # 텍스트 이미지 회전
                                rotated_text = text_img.rotate(-avg_angle, expand=True, resample=Image.BICUBIC)
                                r_w, r_h = rotated_text.size

                                # 메인 이미지를 PIL 형식으로 변환
                                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                                # 회전된 텍스트를 붙여넣을 위치 계산
                                paste_x = center_x - r_w // 2
                                paste_y = center_y - r_h // 2
                                
                                # 텍스트 이미지를 메인 이미지에 붙여넣기 (알파 블렌딩)
                                pil_img.paste(rotated_text, (paste_x, paste_y), rotated_text)
                                
                                # 이미지를 다시 OpenCV 형식으로 변환
                                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                            except Exception as e:
                                print(f"폰트 렌더링 오류: {e}")


            # --- 기존 손 랜드마크 그리기 ---
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 결과 이미지 표시
        cv2.imshow('Hand and Gaze Tracking with Glasses', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
