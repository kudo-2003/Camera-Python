import cv2 
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(2)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1, circle_radius = 1),
                connection_drawing_spec = mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 1, circle_radius = 1)
            )
    cv2.imshow('Camera Face', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()