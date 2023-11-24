#pip install opencv-python
#pip install --upgrade easyocr
# 필요한 라이브러리를 임포트합니다.
import easyocr
print('임포트 함====')
import pandas as pd
import easyocr

from PIL import Image
#태국어 전처리 1
#윤정훈
import os
#pip install pillow
import cv2
import numpy as np



# 엑셀 파일 읽기
# 되는거 전처리 후
# df = pd.read_excel('D:/data/태국어/태국어_전처리.xlsx')
# df = pd.read_excel('D:/data/태국어/태국어_전처리.xlsx')
file_path = 'D:/data/ocr/1107/태국어_전처리_with_result_updated.xlsx'  # 엑셀 파일 경로
df = pd.read_excel(file_path)



ks_list = [1, 3, 5, 7, 9]
ks_list = [1 ]
# ocr_text_열명 = '태국어_전처리_이진화_미디안필터_ocr_text'
# result_열명 = '태국어_전처리_이진화_미디안필터_result'
ocr_text_열명 = '태국어_ocr_text'
result_열명 = '태국어_result'
ocr_text_열명 = '태국어_전처리_미디안필터_kernel_7_ocr_text'
result_열명 = '태국어_전처리_미디안필터_kernel_7_result'

# workbook = openpyxl.load_workbook(file_path)
# worksheet = workbook.active  # 또는 다른 워크시트를 선택하세요

태국어_suc_cnt = 0
태국어_fail_cnt = 0
# OCR 초기화
# reader = easyocr.Reader(['th'])
# reader = easyocr.Reader(['th', 'en'])


custom_model_directory = 'C:/Users/TAMSystech/.EasyOCR/user_network'
custom_model_name = 'thai.pth'
print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ocr_test.py custom_model_directory : {custom_model_directory}')

#reader = easyocr.Reader(['th', 'en'], model_storage_directory=custom_model_directory)

user_network_directory = r'C:/Users/TAMSystech/.EasyOCR/user_network'
#reader = easyocr.Reader(['th', 'en'], recog_network='None_VGG_BiLSTM_CTC')
reader = easyocr.Reader(['th', 'en'] )
#reader = easyocr.Reader(['es', 'en'])
#reader = easyocr.Reader(['th', 'en'], model_storage_directory = user_network_directory, recog_network='None_VGG_BiLSTM_CTC')


#reader = easyocr.Reader(['th', 'en'], recog_network=os.path.join(user_network_directory, 'None-VGG-BiLSTM-CTC.yaml'))

print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ocr_test.py reader : {reader}')
# 모델 초기화
#reader = easyocr.Reader(['th', 'en'], recog_network='None-VGG-BiLSTM-CTC')
#reader = Reader(['en'], recog_network='custom_model.yaml' )
kernel_size_option = 0

# 이미지 디렉토리 설정
image_dir = 'C:/Users/TAMSystech/yjh/img/태국어'
image_dir = 'C:/Users/TAMSystech/yjh/img/백업/1017/thai_img'
image_dir = 'C:/Users/TAMSystech/yjh/img/백업/1017/test'

for ks in ks_list:

    태국어_suc_cnt = 0
    태국어_fail_cnt = 0

    # 이미지 루프
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):  # 혹은 다른 이미지 확장자를 사용하시면 변경해주십시오.

            image_path = os.path.join(image_dir, filename)

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ocr_test 루프 image_path  : {image_path}')


            image = Image.open(image_path)
            # 이미지에서 텍스트 인식
            # results = reader.readtext(image_path)
            # 이미지를 그레이스케일로 변환
            img_cv_grey = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
            print(img_cv_grey.shape)  # 이미지의 차원 확인
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ocr test img_cv_grey.shape : {img_cv_grey.shape}')

            # 한국어는 침식 사용
            # 언어마다 전처리 다 다름
            # 정확도 높은 전처리를 선택할 예정

            # 이진화 적용
            # _, img_cv_bin = cv2.threshold(img_cv_grey, 128, 255, cv2.THRESH_BINARY)
            # for ks in ks_list:
            ocr_text_열명 = f'태국어_전처리_미디안필터_kernel_{ks}_ocr_text'
            result_열명 = f'태국어_전처리_미디안필터_kernel_{ks}_result'
            # 이미지에 미디안 필터 적용

            img_cv_filtered = cv2.medianBlur(img_cv_grey, ks)  # 숫자는 커널 크기, 조절 가능
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ocr test img_cv_filtered : {img_cv_filtered }')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ocr test results 여기서 에러 img_cv_filtered.shape : {img_cv_filtered.shape}')
            # 작은 커널 크기(3x3)를 사용하여 OCR 전처리를 수행
            results = reader.readtext(img_cv_filtered)
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ocr results = reader.readtext(img_cv_filtered) : {results}')

            # OCR 결과를 저장할 리스트
            recognized_words = []
            # OCR 결과에서 bbox를 사용하여 단어들을 위치에 따라 정렬
            results.sort(key=lambda x: x[0][0][0])  # 결과를 x 좌표를 기준으로 정렬

            # 정렬된 결과를 출력
            for (bbox, text, prob) in results:
                recognized_words.append(text)

            # OCR 결과 리스트를 문자열로 결합
            recognized_text = ' '.join(recognized_words)
            print(f'정렬된 결과를 출력 filename : {filename}')
            print(f'정렬된 결과를 출력 recognized_words : {recognized_words}')
            print(f'정렬된 결과를 출력 recognized_text : {recognized_text}')

            # OCR 결과 순회
            # 파일명에서 확장자 제거, 좌우 공백 및 마침표 제거
            filename = os.path.splitext(filename)[0].strip().replace('.', '')
            print(f'파일명에서 확장자 제거, 좌우 공백 및 마침표 제거 filename : {filename}')
            # filename과 같은 태국어 열의 인덱스를 찾습니다.
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ocr test df : {df}')
            idx = df.index[df['태국어'] == filename].tolist()

            if not idx:
                print(f'같은 열 없음 filename : {filename}')
            else:
                print(f'========같은 열 있음 idx : {idx}')
                print(f'========같은 열 있음 image filename : {filename}')
                print(f'========같은 열 있음 recognized_text : {recognized_text}')

                # 추출된 텍스트를 '태국어_ocr_text' 열에 넣습니다.
                df.at[idx[0], ocr_text_열명] = recognized_text
                #print(f'========같은 열 있음 recognized_text : {recognized_text}')

                # OCR 결과를 업데이트

                if recognized_text == df.at[idx[0], '태국어']:
                    df.at[idx[0], result_열명] = 'suc'
                    태국어_suc_cnt += 1
                    print(f'#################같은 열 있음 성공   태국어_suc_cnt : {태국어_suc_cnt}')

                else:
                    df.at[idx[0], result_열명] = 'fail'
                    태국어_fail_cnt += 1
                    print(f'#################fail 같은 열 있음  태국어_fail_cnt : {태국어_fail_cnt}')

            continue  # 일치하는 항목이 없는 경우 다음 이미지로 이동

    last_row_index = df.index[-1]  # 가장 마지막 행의 인덱스
    next_row_index = last_row_index + 1  # 다음 행의 인덱스
    print(f"last_row_index: {last_row_index}")
    print(f"next_row_index: {next_row_index}")
    태국어_tot_cnt = 태국어_suc_cnt + 태국어_fail_cnt

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!태국어_suc_cnt : {태국어_suc_cnt}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!태국어_fail_cnt : {태국어_fail_cnt}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!태국어_tot_cnt : {태국어_tot_cnt}')

    #태국어_성공률 = round(태국어_suc_cnt / 태국어_tot_cnt, 2)
    if 태국어_suc_cnt != 0:
        태국어_성공률 = round(태국어_suc_cnt / 태국어_tot_cnt, 2)
    else:
        태국어_성공률 = 0.00  # 또는 다른 값을 지정해도 됨
    # 소수 셋째 자리에서 반올림하여 나타냅니다.
    # 태국어_tot_cnt = round(태국어_tot_cnt, 2)
    # 다음 행에 '태국어_전처리1_result' 열에 값을 설정
    # df.at[next_row_index, result_열명] = 태국어_suc_cnt
    df.at[next_row_index, result_열명] = f'{태국어_suc_cnt} / {태국어_tot_cnt} ( {태국어_성공률} )'

    print(f"태국어_suc_cnt:@@@@@@@@@@@@@@@@@@@@@@@@@ {태국어_suc_cnt}")
    df.to_excel('D:/data/ocr/1107/태국어_전처리_with_result_updated.xlsx', index=False)

print(f"일치하는 항목 수: {태국어_suc_cnt}")


