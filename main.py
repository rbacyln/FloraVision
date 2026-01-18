import cv2
import mediapipe as mp
import numpy as np
import os

# Settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# Camera Setup (Default camera is usually 0)
cap = cv2.VideoCapture(0)
WINDOW_NAME = "FloraVision"

# Window Settings
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Transparent Overlay Function
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    try:
        bg_h, bg_w, _ = background.shape
        if overlay_size is not None:
            h, w, c = overlay.shape
            if w == 0 or h == 0: return background
            aspect_ratio = h / w
            new_w = overlay_size[0]
            new_h = int(new_w * aspect_ratio)
            overlay = cv2.resize(overlay, (new_w, new_h))

        h, w, c = overlay.shape
        if x + w > bg_w: w = bg_w - x
        if y + h > bg_h: h = bg_h - y
        if x < 0: w = w + x; x = 0
        if y < 0: h = h + y; y = 0
        if w <= 0 or h <= 0: return background
        
        overlay = overlay[:h, :w]
        
        if c < 4: 
            background[y:y+h, x:x+w] = overlay
            return background
            
        overlay_img = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0
        
        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
        return background
    except Exception as e:
        return background

# Load Assets 
img_butterfly = img_bird = img_bouquet = img_crown = None

try:
    if os.path.exists("butterfly.png"): img_butterfly = cv2.imread("butterfly.png", cv2.IMREAD_UNCHANGED)
    if os.path.exists("bird.png"): img_bird = cv2.imread("bird.png", cv2.IMREAD_UNCHANGED)
    if os.path.exists("bouquet.png"): img_bouquet = cv2.imread("bouquet.png", cv2.IMREAD_UNCHANGED)
    if os.path.exists("crown.png"): img_crown = cv2.imread("crown.png", cv2.IMREAD_UNCHANGED)
    print("Assets loaded successfully.")
except Exception as e:
    print("Error loading assets:", e)

# Finger Counting Logic 
def count_fingers(lm_list, hand_label):
    fingers = []
    # Thumb logic depends on whether it's the right or left hand
    if hand_label == "Right": 
        if lm_list[4][1] < lm_list[3][1]: fingers.append(1)
        else: fingers.append(0)
    else: 
        if lm_list[4][1] > lm_list[3][1]: fingers.append(1)
        else: fingers.append(0)
    
    # Other 4 fingers
    for id in [8, 12, 16, 20]:
        if lm_list[id][2] < lm_list[id - 2][2]: fingers.append(1)
        else: fingers.append(0)
            
    return fingers.count(1)

# Main Loop 
while True:
    success, img = cap.read()
    if not success: break
    
    # Resize image for better performance and window fit
    img = cv2.resize(img, (960, 540))
    
    # Flip image for mirror effect
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results_hands = hands.process(img_rgb)
    results_face = face_detection.process(img_rgb)
    
    h_img, w_img, c_img = img.shape
    total_fingers = 0 

    if results_hands.multi_hand_landmarks:
        for i, hand_lms in enumerate(results_hands.multi_hand_landmarks):
            # MediaPipe returns "Right" or "Left" based on the mirrored image
            hand_label = results_hands.multi_handedness[i].classification[0].label
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                lm_list.append([id, int(lm.x * w_img), int(lm.y * h_img)])

            total_fingers = count_fingers(lm_list, hand_label)

            # AR Visuals 
            
            # 1 Finger -> Butterfly
            if total_fingers == 1 and img_butterfly is not None:
                tip_x, tip_y = lm_list[8][1], lm_list[8][2]
                img = overlay_transparent(img, img_butterfly, tip_x - 40, tip_y - 70, (80, 80))

            # 2 Fingers -> Bird
            elif total_fingers == 2 and img_bird is not None:
                tip_x, tip_y = lm_list[8][1], lm_list[8][2]
                img = overlay_transparent(img, img_bird, tip_x - 60, tip_y - 80, (120, 120))

            # 3 Fingers -> Bouquet
            elif total_fingers == 3 and img_bouquet is not None:
                hand_center_x, hand_center_y = lm_list[9][1], lm_list[9][2]
                bouquet_size = 250 
                pos_x = hand_center_x - int(bouquet_size / 2)
                pos_y = hand_center_y - int(bouquet_size * 0.8)
                img = overlay_transparent(img, img_bouquet, pos_x, pos_y, (bouquet_size, bouquet_size))

            # 5 Fingers -> Crown
            elif total_fingers == 5 and img_crown is not None:
                if results_face.detections:
                    for detection in results_face.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x, y, w, h = int(bboxC.xmin * w_img), int(bboxC.ymin * h_img), \
                                     int(bboxC.width * w_img), int(bboxC.height * h_img)
                        
                        crown_width = int(w * 1.5)
                        crown_x = x - int((crown_width - w) / 2)
                        crown_y = y - int(crown_width * 0.8) 

                        img = overlay_transparent(img, img_crown, crown_x, crown_y, (crown_width, 100))
                        break

            # Debug Info
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            text_pos_y = 70 if hand_label == "Right" else 120
            cv2.putText(img, f"{hand_label}: {total_fingers}", (10, text_pos_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow(WINDOW_NAME, img)
    
    # Quit Logic
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()