import cv2
from text_detection_model import TextDetectionModel  # 디텍션 모델 불러오기

# 디텍션 모델 초기화
text_detection_model = TextDetectionModel()  # 예시: TextDetectionModel은 디텍션 모델 클래스입니다.

# LMDB 파일 초기화 또는 열기 (이미지 데이터 및 라벨을 저장하기 위한 데이터베이스)
lmdb_file = "path/to/your/output.lmdb"
lmdb_env = init_or_open_lmdb(lmdb_file)  # 예시: LMDB 파일 초기화 또는 열기 함수

# 원시 이미지 로드 (이미지 경로, 또는 딥 프로젝트의 입력 데이터 형태에 따라)
raw_image = cv2.imread("path/to/your/raw_image.jpg")

# 텍스트 영역 검출
text_boxes = text_detection_model.detect_text(raw_image)

# 각 텍스트 영역에 대해 이미지 추출 및 LMDB에 추가
for i, box in enumerate(text_boxes):
    x, y, w, h = box
    cropped_text_image = raw_image[y:y+h, x:x+w]

    # 이미지 저장 또는 LMDB에 추가
    save_or_add_to_lmdb(cropped_text_image, label=f"image_{i}", lmdb_env)

# LMDB 파일 닫기 (필요한 경우)
lmdb_env.close()

