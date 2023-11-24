import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from custom.models.model import CustomModel
from test import validation

import sys
import importlib
import datetime
import yaml
# 모듈을 다시 로드
importlib.reload(sys)

#reload(sys)
#sys.setdefaultencoding('utf-8')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -*- coding: utf-8 -*-
from PIL import Image


def train(opt):
    global original_log_file_name
    cnt = 0

    current_prediction_accuracy_list = []
    current_date = datetime.datetime.now().strftime("%m-%d")

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!current_date : {current_date}')

    #path_to_weights = fr"C:\Users\TAMSystech\yjh\ipynb\deep-text-recognition-benchmark\saved_models\None-VGG-BiLSTM-CTC-Seed1111\{current_date}"

    # 현재 날짜를 가져옵니다.
    current_date = datetime.datetime.now().strftime("%m-%d")
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!current_date : {current_date}')



    # 원래 경로를 설정합니다.
    original_path_to_weights = fr"C:\Users\TAMSystech\yjh\ipynb\deep-text-recognition-benchmark\saved_models\None-VGG-BiLSTM-CTC-Seed1111\{current_date}"
    #original_path_to_weights = fr"C:\Users\TAMSystech\yjh\ipynb\deep-text-recognition-benchmark\saved_models\TPS-ResNet-BiLSTM-Attn-Seed1111\{current_date}"
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!original_path_to_weights : {original_path_to_weights}')

    # # 경로가 존재하지 않거나 해당 경로에 아무 파일도 없는 경우를 확인합니다.
    # while not os.path.exists(original_path_to_weights) or not os.listdir(original_path_to_weights):
    #
    #     # 현재 날짜를 datetime 객체로 파싱합니다.
    #     current_datetime = datetime.datetime.strptime(current_date, "%m-%d")
    #
    #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!경로가 존재할 때까지 전날로 변경합니다 current_datetime : {current_datetime}')
    #
    #     # 하루를 빼서 전날 날짜를 가져옵니다.
    #     previous_datetime = current_datetime - datetime.timedelta(days=1)
    #
    #     # 전날 날짜를 문자열로 변환합니다.
    #     current_date = previous_datetime.strftime("%m-%d")
    #
    #     # 경로를 업데이트합니다.
    #     original_path_to_weights = fr"C:\Users\TAMSystech\yjh\ipynb\deep-text-recognition-benchmark\saved_models\None-VGG-BiLSTM-CTC-Seed1111\{current_date}"

    # 경로가 존재하면 업데이트합니다.
    path_to_weights = original_path_to_weights
    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!path_to_weights : {path_to_weights}')


    best_accuracy = -1
    current_accuracy_list = []
    for root, dirs, files in os.walk(path_to_weights):

        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 학습된 모델 경로 가져오기 walk path_to_weights : {path_to_weights}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 학습된 모델 경로 가져오기 walk root : {root}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 학습된 모델 경로 가져오기 walk dirs : {dirs}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 학습된 모델 경로 가져오기 walk files : {files}')

        for file in files:
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 학습된 모델 경로 가져오기 for file in files files: : {files:}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 학습된 모델 경로 가져오기 for file in files file : {file}')


            # if "best_prediction_accuracy_" in file:
            if "best_prediction_accuracy_" in file:
                #print("Found matching file:", os.path.join(root, file))

                # 파일 경로에서 파일 이름 추출
                file_name = os.path.join(root, file)

                #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!file_name : {file_name}')

                # 파일 이름을 "."으로 분할하여 파일 이름 부분과 확장자 부분 분리
                file_parts = file_name.split(".")
                #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!file_parts : {file_parts}')

                connected_parts = '.'.join([file_parts[0], file_parts[1]])
                #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!connected_parts : {connected_parts}')

                # current_accuracy = connected_parts.split("_")[-1]
                # print(
                #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!connected_parts.split("_")  : {connected_parts.split("_") }')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!connected_parts.split("_")[-1] : {connected_parts.split("_")[-2]}')
                # print(
                #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!connected_parts.split("_")[-1].split("_") : {connected_parts.split("_")[-2].split("_")}')
                current_prediction_accuracy = connected_parts.split("_")[-2].split("_")[0]
                current_prediction_accuracy = connected_parts.split("_")[-2]

                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!current_prediction_accuracy : {current_prediction_accuracy}')
                #
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!이터 스플릿해서 이 current_accuracy : {current_prediction_accuracy}')

                current_prediction_accuracy = float(current_prediction_accuracy)
                #print(current_prediction_accuracy)

                current_prediction_accuracy_list.append(float(current_prediction_accuracy))
                # 파일 이름 부분을 "_"으로 분할하여 가장 마지막 요소 추출

    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!current_prediction_accuracy_list : {current_prediction_accuracy_list}')
    max_accuracy = max(current_prediction_accuracy_list)
    print("가장 큰 정확도:", max_accuracy)

    max_accuracy_str = str(max_accuracy)
    print("가장 큰 정확도 (문자열):", max_accuracy_str)

    print("path_to_weights:", path_to_weights)

    max_accuracy_file_path = ''
    for root, dirs, files in os.walk(path_to_weights):
        for file in files:
            if max_accuracy_str in file:
            #if "max_accuracy_str" in file or "best_prediction_accuracy_" in file:
                # Your code here

                max_accuracy_file_path = os.path.join(root, file)
                #print("max_accuracy_file_path:", max_accuracy_file_path)
                #print("max_accuracy_str file:", os.path.join(root, file))
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!max_accuracy_file_path : {max_accuracy_file_path}')
    # 기존 모델 불러오기
    opt.saved_model = max_accuracy_file_path
    #opt.saved_model = 'opt.saved_model : C:\Users\TAMSystech\yjh\ipynb\deep-text-recognition-benchmark\saved_models\None-VGG-BiLSTM-CTC-Seed1111\11-23\best_prediction_accuracy_2023-11-23_16-49-53_0.3793_1000.pth'
    #opt.saved_model = 'opt.saved_model : C:\Users\TAMSystech\yjh\ipynb\deep-text-recognition-benchmark\saved_models\None-VGG-BiLSTM-CTC-Seed1111\11-23\best_prediction_accuracy_원본_2023-11-17_17-04-20_0.2226_2.pth'
    #file = 'best_prediction_accuracy_원본_2023-11-17_17-04-20_0.2226_2.pth'
    opt.saved_model = r'C:\Users\TAMSystech\yjh\ipynb\deep-text-recognition-benchmark\saved_models\None-VGG-BiLSTM-CTC-Seed1111\11-23\best_prediction_accuracy_원본_2023-11-17_17-04-20_0.2226_2.pth'

    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! train.py 기존 모델 불러오기 opt.saved_model : {opt.saved_model}')


    #opt.FT = True
    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!63부터 시작하라고 ㅜ 왜 0부터 시작하냐고  saved_model : {opt.saved_model}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.FT : {opt.FT}')
    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!max_accuracy_file_path : {max_accuracy_file_path}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.saved_model : {opt.saved_model}')
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130
    print(f'트레인 처음 들어감!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train opt : {opt}')
    if 'ๆ' in opt.character:
        # 'ๆ' 문자가 있을 경우
        print('트레인 처음 들어감 있음')
    else:
        # 'ๆ' 문자가 없을 경우
        print('트레인 처음 들어감 없음')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 처음 들어와서 Batch_Balanced_Dataset 에 opt 넘겨줄때 opt : {opt}')
    if 'ๆ' in opt.character:
        # 'ๆ' 문자가 있을 경우
        print('트레인 처음 들어와서 Batch_Balanced_Dataset 에 opt 넘겨줄때 opt 들어감 있음')
    else:
        # 'ๆ' 문자가 없을 경우
        print('트레인 처음 들어와서 Batch_Balanced_Dataset 에 opt 넘겨줄때 opt 들어감 false')
    train_dataset = Batch_Balanced_Dataset(opt)

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 트레인  Batch_Balanced_Dataset -> train_dataset : {train_dataset}')


    # print(f'@@@@@@@@@@@@@@@@@@@@train_dataset = Batch_Balanced_Dataset(opt) 에서 opt : {opt}')
    # print(f'@@@@@@@@@@@@@@@@@@@@train_dataset : {train_dataset}')
    current_date = datetime.datetime.now().strftime("%m-%d")

    # 원하는 폴더 경로 생성
    directory = f'./saved_models/{opt.exp_name}/{current_date}/'

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 파일 열기
    log = open(f'{directory}/log_dataset.txt', 'a')

    #log = open(f'./saved_models/{opt.exp_name}/{current_date}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    # valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)

    selected_d = 'JH'
    selected_d = 'JH2'
    selected_d = 'JH3'
    selected_d = 'th'
    #selected_d = 'ttf'
    # print(f'==============밸리드도 나는 트레인으로 일단 하기로 selected_d : {selected_d}')

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 opt.train_dat : {opt.train_data}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 opt.valid_data : {opt.valid_data}')

    #valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt, select_data=[selected_d])


    print(f'###############valid_datasethierarchical_dataset  트레인이랑 같다니까? : {valid_dataset}')
    print(f'###############valid_dataset hierarchical_dataset 트레인이랑 같다니까? len(valid_dataset) : {len(valid_dataset)}')
    print(f'###################valid_dataset_log hierarchical_dataset 트레인이랑 같다니까? : {valid_dataset_log}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!valid_dataset opt.batch_size : {opt.batch_size}')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!valid_loader opt.batch_size : {opt.batch_size}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!valid_loader len(valid_dataset) : {len(valid_dataset)}')
    # 데이터셋 객체에서 라벨과 라벨 수 확인
    # valid_dataset에서 라벨 가져오기


    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Number of Labels in valid_dataset len(labels2) : {len(labels2)}')
    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Number of Labels in valid_dataset : {labels2}')

    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!valid_dataset_log : {valid_dataset_log}')

    print(f'트레인 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train opt.character : {opt.character}')
    if 'ๆ' in opt.character:
        # 'ๆ' 문자가 있을 경우
        print('트레인 함수 들어옴 있음')
    else:
        # 'ๆ' 문자가 없을 경우
        print('트레인 함수 들어옴 없음')

    # text = "태국어 텍스트"
    # encoded_text = text.encode("utf-8")  # 다른 인코딩으로 변환하려면 해당 인코딩을 지정

    text = "태국어 텍스트"
    #opt.character = opt.character.encode("utf-8")  # 다른 인코딩으로 변환하려면 해당 인코딩을 지정
    print(f'##########################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train  인코딩 잘됨???? utf-8 로 인코딩 하면 이상하게 xb8 이런식으로 들어감 이젠 주석했으니 정상 일것 opt.character : {opt.character}')

    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 함수 들왓 CTC 인 경우 opt.character 를 CTCLabelConverter 함수에 넘겨준다 opt.character : {opt.character}')
            if 'ๆ' in opt.character:
                # 'ๆ' 문자가 있을 경우
                print('트레인 함수 들왓 CTC 인 경우 opt.character 를 CTCLabelConverter 함수에 넘겨준다 opt.character 있음')
            else:
                # 'ๆ' 문자가 없을 경우
                print('트레인 함수 들왓 CTC 인 경우 opt.character 를 CTCLabelConverter 함수에 넘겨준다 opt.character 없음')

            print(f'!!!!!!!!!!!!!!!CTCLabelConverter 호출전 len(opt.character) : {len(opt.character)}')
            print(f'!!!!!!!!!!!!!!!CTCLabelConverter 호출전 cnt : {cnt}')
            converter = CTCLabelConverter(opt.character)
            cnt = cnt + 1
            print(f'!!!!!!!!!!!!!!!CTCLabelConverter 호출 완료 cnt: {cnt}')
    else:
        converter = AttnLabelConverter(opt.character)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try.py align with Attention.forward converter : {converter}')

    # 이부분 레이블 수로 수정해야함
    opt.num_class = len(converter.character)
    opt.batch_max_length = opt.num_class
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인.py opt.num_class 1 증가한다 [CTCblank] (  : {opt.num_class}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인.py opt.batch_max_length num_class ( 178 + 1 ) 로 수정  ( preds 에 178 보다 큰 수 나오면 에러남 ( dict 에 없음 ) : {opt.batch_max_length}')

    opt.FT = True

    if opt.rgb:
        opt.input_channel = 3
    #model = Model(opt)
    # 모델 가져오기
    #model = Model(opt)
    # 가중치 파일을 로드
    # checkpoint = torch.load(max_accuracy_file_path)
    #
    # # 모델에 가중치 적용
    # model.load_state_dict(checkpoint)

    print(f'===== 트레인 모델의 아키텍처 ===== opt.FT : {opt.FT}')
    print(f'===== 트레인 모델의 아키텍처 ===== opt.saved_model : {opt.saved_model}')
    #print(f'===== 트레인 현재 모델의 아키텍처 =====처음 생성시 model : {model}')

    # 저장된 모델 파일 경로
    model_path = "your_model.pth"

    # GPU로 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'===== 트레인 모델의 아키텍처 ===== device : {device}')
    # 저장된 모델 불러오기
    #saved_model = torch.load(opt.saved_model, map_location=device)

    # 저장된 모델 불러오기
    #saved_model = torch.load(opt.saved_model)
    print("\n모델 프린트 너무 길어서 생략===== 저장된 모델의 아키텍처 =====")
    #print(saved_model)
    # for key, value in saved_model.items():
    #     print(f"모델 프린트 너무 길어서 생략===== 저장된 모델의 아키텍처 Key: {key}, Shape: {value.shape if isinstance(value, torch.Tensor) else 'Not a Tensor'}")

    #model = Model(opt)  # YourModelClass는 실제 사용하는 모델 클래스로 바꿔야 합니다.

    # 저장된 모델의 가중치를 모델에 로드
    #model.load_state_dict(saved_model['model_state_dict'])
    checkpoint = torch.load(max_accuracy_file_path)

    # 모델에 가중치 적용
    # model.load_state_dict(checkpoint)
    #
    # # 모델을 지정된 장치로 이동
    # model = model.to(device)

    # for key, value in saved_model.items():
    #     print(f"모델을 지정된 장치로 이동===== 저장된 모델의 아키텍처 Key: {key}, Shape: {value.shape if isinstance(value, torch.Tensor) else 'Not a Tensor'}")

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.input_channel : {opt.input_channel}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.output_channel : {opt.output_channel}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.hidden_size : {opt.hidden_size}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.num_class : {opt.num_class}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.batch_max_length : {opt.batch_max_length}')

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.Transformation : {opt.Transformation}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.FeatureExtraction : {opt.FeatureExtraction}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.SequenceModeling : {opt.SequenceModeling}')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.Prediction : {opt.Prediction}')


    # # weight initialization
    # for name, param in model.named_parameters():
    #     if 'localization_fc2' in name:
    #         print(f'Skip {name} as it is already initialized')
    #         continue
    #     try:
    #         if 'bias' in name:
    #             init.constant_(param, 0.0)
    #         elif 'weight' in name:
    #             init.kaiming_normal_(param)
    #     except Exception as e:  # for batchnorm.
    #         if 'weight' in name:
    #             param.data.fill_(1)
    #         continue

    # data parallel for multi-GPU
    # model = torch.nn.DataParallel(model).to(device)
    # model.train()

    # 저장된 모델 가중치 불러오기
    #saved_model = torch.load('path/to/your/saved_model.pth')
    #model.load_state_dict(saved_model['model_state_dict'])
    checkpoint = torch.load(max_accuracy_file_path)

    # 모델에 가중치 적용
    #model.load_state_dict(checkpoint)

    # 키와 값 출력
    # for key, value in saved_model.items():
    #     print(f'Key: {key}, Value Shape: {value.shape}')

    # 모델을 데이터 병렬로 변환하고 지정된 장치로 이동
    # model = torch.nn.DataParallel(saved_model).to(device)
    # model.train()

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.saved_model : {opt.saved_model}')


    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.FT 로 일단 지정 원래 easyocr 예측값 보려고 : {opt.FT}')
    # if opt.saved_model != '':
    #
    #
    #
    #     model = Model(opt)
    #
    #     print(f'loading pretrained model from {opt.saved_model}')
    #     # print(f'===== 트레인 모델의 아키텍처 ===== opt : {opt}')
    #     # print(f'===== 트레인 모델의 아키텍처 ===== opt.FT : {opt.FT}')
    #     print(f'===== 트레인 현재 모델의 아키텍처 =====model : {model}')
    #
    #     if opt.FT:
    #
    #         # print(f'===== 트레인 모델의 아키텍처 ===== opt.FT : {opt.FT}')
    #         # print(f'===== 트레인 현재 모델의 아키텍처 =====model : {model}')
    #
    #         # 저장된 모델 불러오기
    #         saved_model = torch.load(opt.saved_model)
    #         # print("\n===== 저장된 모델의 아키텍처 =====")
    #         # print(saved_model)
    #
    #
    #         model.load_state_dict(torch.load(opt.saved_model), strict=False)
    #     else:
    #         print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!else: opt.FT 로 일단 지정 원래 easyocr 예측값 보려고 : {opt.FT}')
    #         model.load_state_dict(torch.load(opt.saved_model))
    # print("Model:")
    #print(model)

    # 원본
    # 가중치 초기화
    # weight initialization
    # for name, param in model.named_parameters():
    #     if 'localization_fc2' in name:
    #         print(f'Skip {name} as it is already initialized')
    #         continue
    #     try:
    #         if 'bias' in name:
    #             init.constant_(param, 0.0)
    #         elif 'weight' in name:
    #             init.kaiming_normal_(param)
    #     except Exception as e:  # for batchnorm.
    #         if 'weight' in name:
    #             param.data.fill_(1)
    #         continue

    if opt.saved_model != '':

        # 마지막 레이어만 가중치를 업데이트
        opt.FT = True

        #model = Model(opt)

        print(f'loading pretrained model from {opt.saved_model}')
        # print(f'===== 트레인 모델의 아키텍처 ===== opt : {opt}')
        # print(f'===== 트레인 모델의 아키텍처 ===== opt.FT : {opt.FT}')
        #print(f'===== 트레인 현재 모델의 아키텍처 =====model : {model}')



        saved_model = torch.load(opt.saved_model)
        #print(f'Loading pretrained model from model ㅊ = {model}')
        print("\n!!!!!!!!!!!모델 프린트 너무 길어서 생략===== 저장된 모델의 아키텍처 =====")
        #model = Model(opt)
        print(f'Loading pretrained model from {opt.saved_model}')
        #print(f'Loading pretrained model from model = Model(opt) model = {model }')

        #print(f'Loading pretrained model from model model = Model(opt)  model = {model}')
        #print(f'Loading pretrained model from model saved_model = {saved_model}')
        #여기부터
        # yaml 파일 읽기
        yaml_path = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/EasyOCR/user_network/custom.yaml'

        with open(yaml_path, 'r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream)

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!내가 만든 모델 opt : {opt}')

        # 모델 초기화
        input_channel = config['network_params']['input_channel']
        output_channel = config['network_params']['output_channel']
        hidden_size = config['network_params']['hidden_size']
        opt.character = config['character_list']
        # len(opt.character) 와 len(converter.character) 둘 중 뭐가 맞는지 확인 필요 ( 기존 모델은 len(opt.character) 로 돼있는데 len(converter.character) 는 train.py 코드에 있음
        #opt.num_class = len(opt.character)
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try opt.num_class = len(opt.character) yaml 파일에서 읽어온 거 opt.num_class : {opt.num_class}')

        #opt.num_class = len(converter.character)
        # 기존 가중치 사용하려고
        #opt.num_class = 177
        print(
            f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try opt.num_class = len(opt.character) len(converter.character) 로 수정후 ( 1이 더해짐 ) opt.num_class : {opt.num_class}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 내가 만든 모델 opt : {opt}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 내가 만든 모델 input_channel : {input_channel}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 내가 만든 모델 output_channel {output_channel}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 내가 만든 모델 hidden_size : {hidden_size}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 내가 만든 모델 opt.character : {opt.character}')
        # #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 내가 만든 모델 len(opt.character) : {len(opt.character)}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 내가 만든 모델 len(opt.character) : {len(opt.character)}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 내가 만든 모델 opt.num_class : {opt.num_class}')

        output_channel = output_channel * 2
        hidden_size =  hidden_size * 2
        # 기존 저장된 모델 num_class 177
        # 학습 시키려는 num_class 98
        # 기존 저장된 모델 num_class + 학습 시키려는 num_class 178 ( '✓' 추가됨 )
        #opt.num_class = 177
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train  내가 만든 모델 output_channel * 2 변경후 output_channel : {output_channel}')
        # print(
        #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train  내가 만든 모델 output_channel * 2 변경후 hidden_size : {hidden_size}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train  내가 만든 모델 output_channel * 2 변경후 opt.num_class : {opt.num_class}')
        #def __init__(self, input_channel, output_channel, hidden_size, num_class):
        #input_channel, output_channel, hidden_size, num_class
        # model = CustomModel(input_channel, output_channel, hidden_size, opt.num_class)
        # print(f'내가 만든 모델 model = {model}')
        #sys.exit()

        # 원본 모델의 가중치 불러오기
        #saved_model_state_dict = torch.load(opt.saved_model)
        # saved_model_state_dict = CustomModel(input_channel, output_channel, hidden_size, opt.num_class - 1)
        # print(f'원본 모델 model = {saved_model_state_dict}')
        #
        # # DataParallel 로 가중치 로드하면 접두사 에러 해결됨
        # saved_model_state_dict = torch.nn.DataParallel(saved_model_state_dict)
        #
        # print(f'try  strict=False  설정 opt.FT : {opt.FT}')
        # saved_model_state_dict.load_state_dict(torch.load(opt.saved_model), strict=True)
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try 원본 saved_model_state_dict : {saved_model_state_dict}')


        # for name, param in saved_model_state_dict.named_parameters():
        #     # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 name: {name}')
        #     # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 param: {param}')
        #     if "Prediction" in name:  # 마지막 레이어에 대해서만 학습
        #         param.requires_grad = True
        #         print(f'원본 가중치 Name: {name}\n, 원본 가중치 Shape ( 178 ): {param.shape}\n, 원본 가중치 data: {param.data}\n, 원본 가중치 data.shape ( 178 개만 불러와야 함 ): {param.data.shape}\n')


        if opt.FT:

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try opt.num_class 179 개 나와야 함 : {opt.num_class}')
            #opt.num_class = 179
            #opt.num_class = 177
            #model = CustomModel(input_channel, output_channel, hidden_size, opt.num_class)
            # output_channel, hidden_size 두배로 늘림
            model = CustomModel(input_channel, output_channel, hidden_size, opt.num_class)
            model = Model(opt)
            #print(f'내가 만든 모델 model = {model}')

            print(f'try  strict=False  설정 opt.FT : {opt.FT}')
            # 출력개수 같은경우 ( 177, 177 ) DataParallel 로 먼저 가중치 로드하면 접두사 에러 해결됨 ( module. 이 접두사로 붙음 ( 병렬처리시 module. 이 접두사로 붙음 )
            #model = torch.nn.DataParallel(model)

            model.load_state_dict(torch.load(opt.saved_model), strict=False)
            #model.load_state_dict(torch.load(opt.saved_model), strict=True)

            # 출력개수 틀릴시 ( 177, 179 ) strict=False 로 먼저 가중치 dict 로드하고 병렬로 이동시키면 에러 안남. ( 순서를 틀리게 해야함. 출력개수 같은경우와 틀린경우에 따라 )
            model = torch.nn.DataParallel(model)

            # for name, param in model.named_parameters():
            #     # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 name: {name}')
            #     # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 param: {param}')
            #     if "Prediction" in name:  # 마지막 레이어에 대해서만 학습
            #         param.requires_grad = True
            #         print(
            #             f'기존 가중치 Name: {name}\n, 기존 가중치 Shape ( 기존 177 + ( 체크 ) + blank => 179 ): {param.shape}\n, 기존 가중치 data: {param.data}\n')
            #
            #     else:
            #         param.requires_grad = False  # 나머지 레이어는 동결

            # 일단 기존 가중치랑 맞추기로 ( 처음에 너무 정확도가 낮음, 처음 트레인시에도 어느 정도 정확도 유지하는지 보려고 )
            # model.load_state_dict(torch.load(opt.saved_model), strict=False)
            #model.load_state_dict(torch.load(opt.saved_model), strict=True)

            # 저장된 모델의 가중치 불러오기
            # saved_model_state_dict = torch.load(opt.saved_model)
            # print(f'저장된 모델의 가중치 saved_model_state_dict = {saved_model_state_dict}')

            # 저장된 모델의 가중치를 모델에 로드
            # model.load_state_dict(saved_model['model_state_dict'])
            #
            # # 모델을 지정된 장치로 이동
            # model = model.to(device)
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! opt.FT model 가중치: {model}')
            # DataParallel 로 가중치 로드하면 접두사 에러 해결됨
            # model = torch.nn.DataParallel(model)
            # model.load_state_dict(torch.load(opt.saved_model))


        else:

            model = Model(opt)

            model.load_state_dict(torch.load(opt.saved_model))

            # 예시로서 model_dict는 기존 모델의 상태 사전이라고 가정합니다.

            model_dict = torch.load(opt.saved_model)

        #DataParallel 로 가중치 로드하면 접두사 에러 해결됨 ( module. 이 접두사로 붙음 ( 병렬처리시 module. 이 접두사로 붙음 )
        # 출력개수 틀릴시 ( 177, 179 ) strict=False 로 가중치 로드하고 병렬로 이동시키면 에러 안남.
        #model = torch.nn.DataParallel(model)

        # # DataParallel 로 가중치 로드하면 접두사 에러 해결됨
        # model = torch.nn.DataParallel(model)

        # # 모델의 각 레이어에 대해 requires_grad 설정
        # for name, param in model.named_parameters():
        #     # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 name: {name}')
        #     # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 param: {param}')
        #     if "Prediction" in name:  # 마지막 레이어에 대해서만 학습
        #         param.requires_grad = True
        #         print(f'기존 가중치 Name: {name}\n, 기존 가중치 Shape: {param.shape}\n, 기존 가중치 data: {param.data}\n')
        #
        #     else:
        #         param.requires_grad = False  # 나머지 레이어는 동결

        # 모델의 각 레이어에 대해 requires_grad 설정
        for name, param in model.named_parameters():
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 name: {name}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 param: {param}')
            if "Prediction" in name:  # 마지막 레이어에 대해서만 학습
                param.requires_grad = True
                #print(f'기존 가중치 Name: {name}\n, 기존 가중치 Shape ( 기존 177 + ( 체크 ) + blank => 179 ): {param.shape}\n, 기존 가중치 data: {param.data}\n')
                print(
                    f'기존 가중치 Name: {name}\n, 기존 가중치 Shape ( 기존 177 + ( 체크 ) + blank => 179 ): {param.shape}\n')

            else:
                param.requires_grad = False  # 나머지 레이어는 동결

        # 특정 부분의 파라미터를 제외한 파라미터 그룹 생성
        # params_to_update = []
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         params_to_update.append(param)

        # 옵티마이저 생성 시, params_to_update를 전달
        # optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # 원본
        # weight initialization
        # for name, param in model.named_parameters():
        #     if 'localization_fc2' in name:
        #         print(f'Skip {name} as it is already initialized')
        #         continue
        #     try:
        #         if 'bias' in name:
        #             init.constant_(param, 0.0)
        #         elif 'weight' in name:
        #             init.kaiming_normal_(param)
        #     except Exception as e:  # for batchnorm.
        #         if 'weight' in name:
        #             param.data.fill_(1)
        #         continue

        # 마지막 레이어의 가중치 가져오기
        #last_layer_weight = model.Prediction.weight.data
        #last_layer_bias = model.Prediction.bias.data
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train last_layer_weight : {last_layer_weight}')
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train  last_layer_bias : {last_layer_bias}')

        # for name, param in model.named_parameters():
        #     if 'localization_fc2' in name:
        #         print(f'Skip {name} as it is already initialized')
        #         continue
        #     try:
        #         if 'bias' in name:
        #             # 기존 가중치로 초기화
        #             param.data.copy_(saved_model_state_dict[name])
        #         elif 'weight' in name:
        #             # 기존 가중치로 초기화
        #             param.data.copy_(saved_model_state_dict[name])
        #     except Exception as e:  # for batchnorm.
        #         if 'weight' in name:
        #             param.data.fill_(1)
        #         continue

        #model.load_state_dict(torch.load(opt.saved_model))

        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,

              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,

              opt.SequenceModeling, opt.Prediction)

        # print(
        #
        #     f'strict !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.input_channel : {opt.input_channel}')
        #
        # print(
        #
        #     f'strict !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.output_channel : {opt.output_channel}')
        #
        # print(f'strict !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.hidden_size : {opt.hidden_size}')
        #
        # print(f'strict !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.num_class : {opt.num_class}')
        #
        # print(
        #
        #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.batch_max_length : {opt.batch_max_length}')
        #
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.Transformation : {opt.Transformation}')
        #
        # print(
        #
        #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.FeatureExtraction : {opt.FeatureExtraction}')
        #
        # print(
        #
        #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.SequenceModeling : {opt.SequenceModeling}')
        #
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!model input parameters opt.Prediction : {opt.Prediction}')

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model cuda 로 보내기전 아키텍쳐: {model}')

        model = model.cuda()
        #model = torch.nn.DataParallel(model).to(device)

        # 모든 파라미터에 대해 초기 설정
        # for param in model.parameters():
        #     param.requires_grad = True  # 모든 파라미터를 업데이트하도록 설정
        # for param in model.parameters():
        #     param.requires_grad = False  # 모든 파라미터를 업데이트하도록 설정

        # 특정 부분의 파라미터에 대해 동결 설정
        # for param in model.some_specific_part.parameters():
        #     param.requires_grad = False  # 해당 부분의 파라미터는 업데이트하지 않음



        model.train()

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model cuda 로 보낸후 아키텍쳐: {model}')


        # 모델의 가중치를 출력
        # for name, param in model.named_parameters():
        #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 name: {name}')
        #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 param: {param}')
        #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 param.data: {param.data}')
        #     if param.requires_grad:
        #         print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져왔음')
        #         #print(name, param.data)
        #         # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 name: {name}')
        #         # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!가중치 잘 가져와야함!!!!!!!!! model 가중치 param.data: {param.data}')

    # 원본
    # 가중치 초기화
    # weight initialization
    # for name, param in model.named_parameters():
    #     if 'localization_fc2' in name:
    #         print(f'Skip {name} as it is already initialized')
    #         continue
    #     try:
    #         if 'bias' in name:
    #             init.constant_(param, 0.0)
    #         elif 'weight' in name:
    #             init.kaiming_normal_(param)
    #     except Exception as e:  # for batchnorm.
    #         if 'weight' in name:
    #             param.data.fill_(1)
    #         continue

    # 원본
    # for name, param in model.named_parameters():
    #     if 'localization_fc2' in name:
    #         print(f'Skip {name} as it is already initialized')
    #         continue
    #     try:
    #         if "Prediction" in name:  # 마지막 레이어에 대해서만 학습
    #
    #             if 'bias' in name:
    #                 init.constant_(param, 0.0)
    #             elif 'weight' in name:
    #                 init.kaiming_normal_(param)
    #
    #     except Exception as e:  # for batchnorm.
    #         if 'weight' in name:
    #             param.data.fill_(1)
    #         continue
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            #if "Prediction" in name:  # 마지막 레이어에 대해서만 학습

            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)

        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # 기존 학습된 가중치 로드
    # pretrained_weights = torch.load(opt.saved_model)
    #
    # # 모델의 각 매개변수에 대해 가중치 초기화 또는 기존 가중치로 설정
    # for name, param in model.named_parameters():
    #     if 'localization_fc2' in name:
    #         print(f'Skip {name} as it is already initialized')
    #         continue
    #
    #     try:
    #         if 'bias' in name:
    #             init.constant_(param, 0.0)
    #         elif 'weight' in name:
    #             # 사용자 정의된 초기화 코드나 기존 가중치 설정 코드를 추가
    #             if name in pretrained_weights:
    #                 param.data = pretrained_weights[name]
    #                 print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try 기존 가중치로 초기화 한다 name : {name}')
    #                 print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try 기존 가중치로 초기화 한다 param.data : {param.data}')
    #
    #             else:
    #                 init.kaiming_normal_(param)
    #     except Exception as e:  # for batchnorm.
    #         if 'weight' in name:
    #             param.data.fill_(1)
    #         continue

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try.py align with Attention.forward criterion : {criterion}')

    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # opt.lr = 0.0085  # 학습률을 더 작게 조정
    # opt.adam = True
    
    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        # 원본
        # adam: False
        # lr: 1
        # beta1: 0.9
        # rho: 0.95
        # eps: 1e-08

        # opt.lr = 0.0085  # 학습률을 더 작게 조정
        # #opt.beta1: 0.9  # Adam의 첫 번째 모멘텀 값
        # opt.rho = 0.9  # Adadelta의 감마 값 (보통 0.9로 설정)
        # opt.eps = 1e-08  # Adadelta의 epsilon 값

        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/{current_date}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            # print(f'~~~~~~~~~~~~~~~~~~~~~이젠 인코딩 k : {k}')

            # print(f'~~~~~~~~~~~~~~~~~~~~~이젠 인코딩 v : {v}')

            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        # print(f'~~~~~~~~~~~~~~~~~~~~~이젠 인코딩 opt_log :  ')

        try:
            # opt_file.write(opt_log)
            # opt_log 변수를 UTF-8로 인코딩
            opt_log_encoded = opt_log.encode('utf-8')
            # print(f'~~~~~~~~~~~~~~~~~~~~~utf-8 로 opt_log_encoded : {opt_log_encoded}')
            # opt_log_encoded = opt_log.encode('utf-8')
            opt_log_decoded = opt_log_encoded.decode('utf-8')
            # print(f'~~~~~~~~~~~~~~~~~~~~~utf-8 로 opt_log_decoded : {opt_log_decoded}')
            opt_file.write(opt_log_decoded)
            # print(f'~~~~~~~~~~~~~~~~~~~~~utf-8 로 opt_log_encoded : {opt_log_encoded}')
            # print('찍은거 성공!!!!!!!!!!!!!!!!!')
        except UnicodeEncodeError:
            # 에러가 발생할 때 대체 문자로 변경
            # print(f'에러가 발생할 때 대체 문자로 변경 :  ')
            # print('opt_log 찍기전encode 가 아예 안되???????????!')
            opt_log = opt_log.encode("utf-8", "ignore").decode("utf-8")
            # print('utf-8 로 하라고!!!!!!!!!!!!!!!!   opt_log 찍기전encode 가 아예 안되???????????!')
            # print(f'@@@@@@@@@@@@utf-8 로 하라고! opt_log : {opt_log}')
            # print('@@@@@@@@@ 프린트는 에러 안나고 .write 함수에서 에러남!@!!!!!!!')
            # opt_file.write(opt_log)
            # print('찍은거 성공!!!!!!!!!!!!!!!!!')

        # print(f'~~~~~~~~~~~~~~~~~~~~~이젠 인코딩 opt_log : {opt_log}')
        # print('opt_log 찍기전!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(opt_log)
        # print('opt_log 찍은후!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!기존 모델 있을 경우!!!!!!!!  opt.saved_model : {opt.saved_model}')
    if opt.saved_model != '':
        try:
            # start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            # 파일 경로에서 파일 이름 추출
            file_name = opt.saved_model
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!기존 모델 있을 경우!!!!!!!!  opt.saved_model : {opt.saved_model}')
            # 파일 이름을 "."으로 분할하여 파일 이름 부분과 확장자 부분 분리
            file_parts = file_name.split(".")

            connected_parts = '.'.join([file_parts[0], file_parts[1]])
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!기존 모델 있을 경우!!!!!!!!  connected_parts : {connected_parts}')
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!기존 모델 있을 경우!!!!!!!!  connected_parts.split("_") : {connected_parts.split("_")}')
            #start_iter = connected_parts.split("_")[-1].split("_")[1]
            start_iter = int(connected_parts.split("_")[-1])
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!기존 모델 있을 경우!!!!!!!!  start_iter : {start_iter}')
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!기존 모델 있을 경우 들어오라고!!!!!!!!!!!!!!!!!!  start_iter : {start_iter}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt saved_model split : {opt.saved_model.split('_')[-1]}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!start_iter : {start_iter}')
            #
            # print(f'continue to train, start_iter: {start_iter}')
            #
            # print(f'트레인 start_iter: {start_iter}')

            print(f'트레인 opt.num_iter: {opt.num_iter}')

            if(start_iter > opt.num_iter):
            # 기존 저장된 가중치 이터레이션이 클 경우 처음부터 로그 저장하기 위해서 start_iter = 0 으로 초기화
                start_iter = 0

                print(f'트레인 if(start_iter < opt.num_iter) 면 0부터 시작 start_iter : {start_iter}')

            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!split : {split}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    best_prediction_accuracy = -1
    # 기존 저장된 가중치 이터레이션이 클 경우 처음부터 로그 저장하기 위해서 start_iter = 0 으로 초기화
    start_iter = 0
    iteration = start_iter


    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.saved_model : {opt.saved_model}')
    #
    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!start_iter : {start_iter}')
    #
    #
    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iteration : {iteration}')

    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!start_iter : {start_iter}')
    #
    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iteration : {iteration}')
    # 모든 결과를 저장할 빈 리스트 만들기
    all_results = []

    # 실제값과 정확도를 저장할 리스트
    #correct_predictions = []
    label_cnt = 0

    while (True):

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 train part opt : {opt}')


        # train part
        image_tensors, labels = train_dataset.get_batch()
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train while (True) image_tensors, labels = train_dataset.get_batch() : len(labels) : {len(labels)}')
        # print(
        #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train while (True) image_tensors, labels = train_dataset.get_batch() : labels : {labels}')

        # 현재 시간을 얻어옵니다.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        image = image_tensors.to(device)
        #print(f'트레인 함수 Batch_Balanced_Dataset 에서 트레인데이터셋 받아와서 labels  : {labels}')
        #'labels' 변수에서 'ๆ' 문자가 있는지 확인
        # if '✓' in labels:
        #     # 'ๆ' 문자가 있을 경우
        #     print('Batch_Balanced_Dataset 에서 트레인데이터셋 받아와서 라벨에도 없음 있음')
        # else:
        #     # 'ๆ' 문자가 없을 경우
        #     print('Batch_Balanced_Dataset 에서 트레인데이터셋 받아와서 라벨에도 없음 없음')

        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인  converter.encode 전 opt.batch_max_length : {opt.batch_max_length}')
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)

        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인  converter.encode 후 len(labels) : {len(labels)}')
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인  converter.encode 후 length : {length}')

        batch_size = image.size(0)

        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!이미지 데이터의 개수 배치사이즈가 192 에서 96 으로 바뀜 batch_size : {batch_size}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!len(labels) : {len(labels)}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iteration : {iteration}')
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!accuracy : {accuracy}')

        # 이미지 텐서를 PIL 이미지로 변환
        # print('@@@@@@@@@@@@@@이미지 텐서를 PIL 이미지로 변환')
        # image_pil = Image.fromarray(image.cpu().numpy()[0][0], 'L')  # 단일 채널 이미지인 경우

        # 이미지 파일로 저장
        # image_pil.save('saved_image.png')  # 이미지를 'saved_image.png' 파일로 저장

        # 이미지 파일 경로 출력
        # print("이미지 파일이 저장된 경로: saved_image.png")


        if 'CTC' in opt.Prediction:
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Prediction if  CTC opt.Prediction: text : {text}')

            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! preds : {preds }')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! preds.shape : {preds.shape}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! len(preds) : {len(preds)}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! batch_size : {batch_size}')
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! preds_size : {preds_size}')
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! text : {text}')
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! preds_size : {preds_size}')
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! length : {length}')

        else:

            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try.py align with Attention.forward :  ')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try.py align with Attention.forward model :  {model}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try.py align with Attention.forward image :  {image}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try.py align with Attention.forward text :  {text}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try.py align with Attention.forward text[:, :-1] :  {text[:, :-1]}')

            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try.py align with Attention.forward cost : {cost}')

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iteration : {iteration}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!valInterval : {opt.valInterval}')
        # validation part
        # print('!!!!!!!!!!!!!!!결과 출력하기 전')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!결과 출력하기 전 iteration 1 증가시키기 전: {iteration}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!결과 출력하기 전 iteration 1 증가시키기 전 마지막도 출력해라!!!!!!  opt.num_iter: {opt.num_iter}')

        #if ( iteration + 1) % opt.valInterval == 0 or iteration == 0:  # To see training progress, we also conduct validation when 'iteration == 0'
        if ( iteration + 1) % opt.valInterval == 0 or iteration == 0 or ( iteration + 1) == opt.num_iter or iteration == start_iter:  # To see training progress, we also conduct validation when 'iteration == 0'

            print('!!!!!!!!!!!!!!!결과 출력한다')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iteration : {iteration}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.valIntervall : {opt.valInterval}')
            print(
                f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iteration + 1) % opt.valInterval : {(iteration + 1) % opt.valInterval}')

            # 결과를 출력하는 코드

            elapsed_time = time.time() - start_time

            log_file = f'./saved_models/{opt.exp_name}/{current_date}/log_train.txt'

            # 현재 시간을 얻어옵니다.
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # 경로 끝에 '_'를 추가하고 현재 시간을 추가한 파일 경로를 생성합니다.
            log_file_with_time = os.path.splitext(log_file)
            log_file_with_time = f'log_train_{log_file_with_time[0]}_{current_time}{log_file_with_time[1]}'

            # 파일 열기
            #with open(log_file_with_time, 'a', encoding='utf-8') as log:


            # for log
            #with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
            #with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a', encoding='utf-8') as log:
            #with open(f'./saved_models/{opt.exp_name}/log_train_{current_time}.txt', 'a', encoding='utf-8') as log:
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!파일 저장전 iteration: {iteration}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!파일 저장전 iteration + 1: {iteration + 1}')

            #         f"./saved_models/{opt.exp_name}/{current_date}/log_train_{current_time}_{iteration + 1}.txt
            with open(f'./saved_models/{opt.exp_name}/{current_date}/log_train_{current_time}_{iteration + 1}.txt', 'a', encoding='utf-8') as log:
                # 나머지 코드
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! validation 함수 넘겨주기전 OPT: {opt}')

                model.eval()
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! validation 함수 넘겨주기전 model.eval() OPT: {opt}')

                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! validation 함수 넘겨주기전 with torch.no_grad() OPT: {opt}')

                model.train()
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! validation 함수 넘겨주기전 model.train() OPT: {opt}')
                #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!validation labels : {labels}')
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!validation 함수에서 라벨 개수가 줄어  len(labels) : {len(labels)}')
                #
                # # train_dataset의 데이터셋 구성 확인
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Train Dataset Information:")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Number of Samples in Train Dataset:", len(train_dataset.get_batch()))

                # unique_labels = set()
                # for _, labels in length:
                #     unique_labels.update(labels)


                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Number of Unique Labels in Train Dataset:",
                #       unique_labels)  # unique_labels을 사용해 고유 라벨 수 확인

                # valid_loader의 데이터셋 구성 확인
                # print("\nValidation Dataset Information:")
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Number of Samples in Validation Dataset:", len(valid_loader.dataset))

                unique_labels = set()
                for _, labels2 in valid_loader.dataset:
                    unique_labels.update(labels2)
                    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!54가 어디서 튀어나옴????for _, labels in valid_loader.dataset  len(labels) : {len(labels2)}')
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!54가 어디서 튀어나옴????for _, labels in valid_loader.dataset  labels 이젠 돼지 2로 바꿨으니깐!!!!!!!!!!!!!!!!!!! : {labels2}')

                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Unique Labels in Validation Dataset:", len(unique_labels))
                # print("Validation Loader Options (opt):", valid_loader.dataset.opt)
                # print("Train Dataset Options (opt):", train_dataset.opt)

                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 2/2 가 안찍혀 opt.num_iter : {opt.num_iter}')
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 2/2 가 안찍혀 iteration : {iteration}')
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 2/2 가 안찍혀 iteration + 1 : {iteration + 1}')
                # training loss and validation loss
                loss_log = f'[{iteration + 1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                # if current_accuracy > best_accuracy:
                #     best_accuracy = current_accuracy
                #     torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                # if current_norm_ED > best_norm_ED:
                #     best_norm_ED = current_norm_ED
                #     torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    # torch.save(model.state_dict(),
                    #            f'./saved_models/{opt.exp_name}/best_accuracy_{current_time}_{current_accuracy:.4f}.pth')
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.exp_name}/{current_date}/best_accuracy_{current_time}_{current_accuracy:.4f}_{iteration + 1}.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    # torch.save(model.state_dict(),
                    #            f'./saved_models/{opt.exp_name}/best_norm_ED_{current_time}_{current_norm_ED:.4f}.pth')
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.exp_name}/{current_date}/best_norm_ED_{current_time}_{current_norm_ED:.4f}_{iteration + 1}.pth')

                print(f"!!!!!!current_accuracy: {current_accuracy}")
                print(f"!!!!!!current_norm_ED: {current_norm_ED:.4f}")

                #best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                correct_predictions = []
                fail_predictions = []
                predicted_result_log = ''
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'

                # gt.txt 에서 실패한 행 읽어와서 저장
                # Read gt.txt file and get failed predictions
                gt_file_path = r'C:\Users\TAMSystech\yjh\ipynb\deep-text-recognition-benchmark\data\ttf\gt\LGBD\태국어\gt.txt'

                failed_gt_predictions = []

                for gt, pred, confidence in zip(labels, preds, confidence_score):
                    label_cnt = label_cnt + 1

                    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label_cnt 갑자기 끝에 라벨 텍스트길이로 바뀌어 버림 ㅜ 개짜증나 ㅜ : {label_cnt}')

                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'

                    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트라이 predicted_result_log 쌓는다 : {predicted_result_log}')

                    is_correct = pred == gt
                    correct_predictions.append(is_correct)

                    if is_correct:
                        # Code for correct prediction
                        pass
                    else:
                        # Code for incorrect prediction
                        # Save gt to a text file
                        #fail_predictions.append(gt)
                        # for 루프 내부에서 fail_predictions에 gt와 pred를 딕셔너리로 추가
                        fail_predictions.append({'gt': gt, 'pred': pred})


                # print(
                #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 fail 로 예측함 잘못 예측한거 리스트에 저장 fail_predictions : {fail_predictions}')

                #fail_predictions_save_dir = r'fail_predictions'

                # with open(f'./saved_models/{opt.exp_name}/{current_date}/fail_predictions_{current_time}_{iteration + 1}.txt',
                #           'w', encoding='utf-8') as fail_predictions_log:
                #
                #     #fail_predictions_log.write(fail_predictions + '\n')
                #     fail_predictions_log.write('\n'.join(map(str, fail_predictions)) + '\n')
                #
                #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트레인 fail_predictions_ 저장 fail_predictions : {fail_predictions}')

                # 실패한 문장 이미지와 함께 텍스트 파일로 저장
                # Save only failed predictions to a text file along with the corresponding lines from gt.txt
                with open(gt_file_path, 'r', encoding='utf-8') as gt_file:
                    lines = gt_file.readlines()

                    for line in lines:

                        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!line : {line}')

                        # Split the line using tab as a separator
                        parts = line.split('\t')
                        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!parts : {parts}')

                        # Check if the part after the tab matches any ground truth in fail_predictions
                        for failed_gt in fail_predictions:
                            t = failed_gt['gt']
                            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트라이 for failed_gt in fail_predictions failed_gt : {failed_gt}')
                            # print(
                            #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트라이 for failed_gt in fail_predictions failed_gt : {t}')

                            #if len(parts) > 1 and parts[1].strip() == failed_gt:
                            if len(parts) > 1 and parts[1].strip() == failed_gt['gt']:
                                #failed_gt_predictions.append(line.strip())  # strip to remove leading/trailing whitespaces
                                failed_gt_predictions.append(f"{line.strip()}\t|\t{failed_gt['pred']}")
                                #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!failed_gt_predictions : {failed_gt_predictions}')

                                # fail_predictions.append(line.strip())  # strip to remove leading/trailing whitespaces
                                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!이미지도 있어야 해 fail_predictions : {fail_predictions}')

                # Save only failed predictions and corresponding lines to a text file
                gt_output_file_path = f'./saved_models/{opt.exp_name}/{current_date}/failed_gt_predictions_{current_time}_{iteration + 1}.txt'
                with open(gt_output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write('\n'.join(failed_gt_predictions) + '\n')

                print(f'실패한 GT 행과 이미지 저장 gt_output_file_path : {gt_output_file_path}')


                #############################################



                predicted_result_log += f'{dashed_line}'


                # 정확도 계산
                accuracy = sum(correct_predictions) / len(correct_predictions)

                #best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}, {"accuracy":17s}: {accuracy:0.2f}'
                #best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                #accuracy_log = f'{"Accuracy":17s}: {accuracy:0.3f}\n{"correct_predictions_len":17s}: {sum(correct_predictions)}\n{"predictions_len":17s}: {len(correct_predictions)}'

                #best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}, {"Aaccuracy":17s}: {accuracy:0.2f}\n'
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}, {"Aaccuracy":10s}({sum(correct_predictions)} / {len(correct_predictions)}): {accuracy:0.2f}\n'
                best_model_log += f'{dashed_line}\n'
                # print(accuracy_log)
                # log.write(accuracy_log + '\n')
                # 정확도 출력
                # accuracy_log = f'{"Accuracy":17s}: {accuracy:0.3f}'
                #accuracy_log = f'{"Accuracy":17s}: {accuracy:0.3f}\n{"correct_predictions_len":17s}: {sum(correct_predictions)}\n{"predictions_len":17s}: {len(correct_predictions)}'
                accuracy_log = f'{"correct_predictions_len":17s}: {sum(correct_predictions)}\n{"predictions_len":17s}: {len(correct_predictions)}\n'
                #accuracy_log += f'{dashed_line}\n'
                # print(accuracy_log)
                #best_model_log += dashed_line
                best_model_log += accuracy_log

                #loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'

                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # print(predicted_result_log)
                # log.write(predicted_result_log + '\n')

                # show some predicted results
                #dashed_line = '-' * 80
                # head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                # predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                # for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                #     if 'Attn' in opt.Prediction:
                #         gt = gt[:gt.find('[s]')]
                #         pred = pred[:pred.find('[s]')]
                #
                #     predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                # predicted_result_log += f'{dashed_line}'
                # print(predicted_result_log)
                # log.write(predicted_result_log + '\n')
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 트레인 labels 길이 : {len(labels)}')

                num_labels = len(labels)
                num_preds = len(preds)
                num_confidence = len(confidence_score)
                # print(f"개짜증나!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! labels: {labels}")
                # print(f"개짜증나!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! validation validationvalidationvalidation Number of labels: {num_labels}")
                # print(f"Number of preds: {num_preds}")
                # print(f"Number of confidence scores: {num_confidence}")
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label_cnt 또 줄기 전 len(labels) : {len(labels)}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label_cnt 또 줄기 전 len(labels) : {len(labels)}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label_cnt 또 줄기 전 len(preds) : {len(preds)}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label_cnt 또 줄기 전 len(confidence_score) : {len(confidence_score)}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label_cnt 갑자기 끝에 라벨 텍스트길이로 바뀌어 버림 ㅜ 개짜증나 ㅜ labels : {labels}')
                # correct_predictions = []
                # for gt, pred, confidence in zip(labels, preds, confidence_score):
                #     label_cnt = label_cnt + 1
                #
                #     #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label_cnt 갑자기 끝에 라벨 텍스트길이로 바뀌어 버림 ㅜ 개짜증나 ㅜ : {label_cnt}')
                #
                #     if 'Attn' in opt.Prediction:
                #         gt = gt[:gt.find('[s]')]
                #         pred = pred[:pred.find('[s]')]
                #
                #     predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                #
                #     is_correct = pred == gt
                #     correct_predictions.append(is_correct)
                #
                #     # result = f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                #     # all_results.append(result)

                #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트라이 correct_predictions 전체 : {correct_predictions}')
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!트라이 correct_predictions 전체 길이 : {len(correct_predictions)}')

                predicted_result_log += f'{dashed_line}'
                #print(predicted_result_log)
                log.write(predicted_result_log + '\n')

                print('=' * 200)


                # for gt, pred, confidence in zip(labels, preds, confidence_score):
                #     if 'Attn' in opt.Prediction:
                #         gt = gt[:gt.find('[s]')]
                #         pred = pred[:pred.find('[s]')]
                #
                #     is_correct = pred == gt
                #     correct_predictions.append(is_correct)
                #
                #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!is_correct : {is_correct}')
                #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!gt : {gt}')
                #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pred : {pred}')
                #
                #     predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{is_correct}\n'


                #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!정확도 계산 len(correct_predictions) : {len(correct_predictions)}')

                # # 정확도 계산
                # accuracy = sum(correct_predictions) / len(correct_predictions)
                # #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!correct_predictions : {correct_predictions}')
                # # 정확도 출력
                # #accuracy_log = f'{"Accuracy":17s}: {accuracy:0.3f}'
                # accuracy_log = f'{"Accuracy":17s}: {accuracy:0.3f}\n{"correct_predictions_len":17s}: {sum(correct_predictions)}\n{"predictions_len":17s}: {len(correct_predictions)}'
                # #print(accuracy_log)
                # log.write(accuracy_log + '\n')

                current_prediction_accuracy = accuracy

                # if current_prediction_accuracy > best_prediction_accuracy:
                #     best_prediction_accuracy = current_prediction_accuracy
                #     torch.save(model.state_dict(),
                #                f'./saved_models/{opt.exp_name}/{current_date}/best_prediction_accuracy_{current_time}_{accuracy:.4f}_{iteration + 1}.pth')
                if current_prediction_accuracy > best_prediction_accuracy:
                    best_prediction_accuracy = current_prediction_accuracy
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.exp_name}/{current_date}/best_prediction_accuracy_{current_time}_{accuracy:.4f}_{iteration + 1}.pth')

                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!로그 다 찍고 저장 !!!!!!!!!!!!!best_prediction_accuracy : {best_prediction_accuracy}')
                print(
                    f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!로그 다 찍고 저장 !!!!!!!!!!!!!current_prediction_accuracy : {current_prediction_accuracy}')

            # 훈련이 끝난 후, 모든 결과 출력
            for result in all_results:
                pass
                #print('훈련 끝남==========')
                #print(result)

        # save model per 1e+5 iter.
        #if (iteration + 1) % 1e+5 == 0:

        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iter 마다 주기적으로 가중치 저장한다 (num_iter // 2) : {(opt.num_iter // 2)}')
        #
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 앞에 날짜 넣어서 저장 !!!!!!!  opt.num_iter : {opt.num_iter}')
        #
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 앞에 날짜 넣어서 저장 !!!!!!!  iteration : {iteration}')

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!current_time : {current_time}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iteration  : {iteration}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.num_iter  : {opt.num_iter}')
        #if (iteration + 1) % (opt.num_iter // 2) == 0:
        if (opt.num_iter != 0 and iteration != 0) and ((opt.num_iter >= 2) and (iteration + 1) % (opt.num_iter // 2) == 0):
            # 해당 코드 블록 실행

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iter 마다 주기적으로 가중치 저장 시작 (num_iter // 2) 저장하라고!!!!!!!!!!: {(opt.num_iter // 2)}')
            torch.save(
                #model.state_dict(), f'./saved_models/{opt.exp_name}/{current_date}/iter_{iteration + 1}.pth')
                model.state_dict(), f'./saved_models/{opt.exp_name}/{current_date}/iter_{current_time}_{accuracy:.4f}_{iteration + 1}.pth')

        # print(f'@@@@@@@@@@@@@@@@@@@@@@@@opt.num_iter : {opt.num_iter}')

        # 원래 파일 경로 및 파일명
        original_log_file_name = f"./saved_models/{opt.exp_name}/{current_date}/log_train_{current_time}_{iteration + 1}.txt"
        #original_log_file_name = f"./saved_models/{opt.exp_name}/{current_date}/log_train_{current_time}_{iteration}.txt"

        # 파일 이름 변경
        #os.rename(original_log_file_name, new_log_file_name)

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!파일의 존재 여부를 확인 original_log_file_name : {original_log_file_name}')

        # 파일의 존재 여부를 확인
        if os.path.exists(original_log_file_name):
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!파일의 존재 여부를 확인 있음___________  original_log_file_name : {original_log_file_name}')

            # 원래 파일 경로 및 파일명
            #original_log_file_name = f"./saved_models/{opt.exp_name}/{current_date}/log_train_{current_time}_{iteration + 1}.txt"

            # 추가할 문자열
            suffix = f"{accuracy:.4f}_{iteration + 1}.txt"

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!각 이터마다 애큐러시 넣어서 파일명 업데이트 해야하는데 처음은 업데이트가 이상하게 안됨  suffix : {suffix}')

            # 새로운 파일 경로 및 파일명 생성
            new_log_file_name = original_log_file_name.replace(f'{iteration + 1}.txt', suffix)

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!새로운 파일 경로 및 파일명 생성 new_log_file_name : {new_log_file_name}')

            # 파일 이름 변경 (original_log_file_name가 존재할 경우에만 실행)
            os.rename(original_log_file_name, new_log_file_name)
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!새로운 파일 경로 및 파일명 생성 new_log_file_name : {new_log_file_name}')

        else:
            #print(f"파일 '{original_log_file_name}'이 존재하지 않습니다.")

            # # 원래 파일 경로 및 파일명
            # original_log_file_name = f"./saved_models/{opt.exp_name}/{current_date}/log_train_{current_time}_{iteration + 1}.txt"
            #
            # # 추가할 문자열
            # suffix = f"{accuracy:.4f}_{iteration + 1}.txt"
            #
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!각 이터마다 애큐러시 넣어서 파일명 업데이트 해야하는데 처음은 업데이트가 이상하게 안됨  suffix : {suffix}')
            #
            # # 새로운 파일 경로 및 파일명 생성
            # new_log_file_name = original_log_file_name.replace(f'{iteration + 1}.txt', suffix)
            #
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!새로운 파일 경로 및 파일명 생성 new_log_file_name : {new_log_file_name}')
            #
            # # 파일 이름 변경 (original_log_file_name가 존재할 경우에만 실행)
            # os.rename(original_log_file_name, new_log_file_name)
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!새로운 파일 경로 및 파일명 생성 new_log_file_name : {new_log_file_name}')

            #####################################

        print(f'!!!!!!!!!!!!opt.num_iter : {opt.num_iter}')
        print(f'!!!!!!!!!!!!iteration + 끝내 : {iteration + 1}')

        #opt.num_iter = 602

        #if (iteration + 1) == opt.num_iter:
        if (iteration + 1) == opt.num_iter or (iteration + 1) >= opt.num_iter:
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iter + 1 이랑  opt.num_iter 랑 같으면 트레인 종료한다 iteration 에 1 더하기 전: {iteration}')
            # print(
            #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iter + 1 이랑  opt.num_iter 랑 같으면 트레인 종료한다 opt.num_iter: {opt.num_iter}')
            print(f'!!!!!!!!!!!!opt.num_iter : {opt.num_iter}')
            print(f'!!!!!!!!!!!!iteration +  : {iteration + 1}')
            #print(f'!!!!!!!!!!!!iteration +  : {iteration + 1}')
            print('end the training')
            start_iter = iteration
            sys.exit()
        iteration += 1

        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 iteration 1 증가시킨다 0에서 1로  iteration : {iteration}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 iteration 1 증가시킨다 0에서 1로 opt.num_iter  : {opt.num_iter}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 iteration 1 증가시킨다 0에서 1로 accuracy : {accuracy:.4f}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 iteration 1 증가시킨다 0에서 1로 accuracy sum / len : {sum(correct_predictions)} / {len(correct_predictions)}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=False, help='path to training dataset')
    parser.add_argument('--valid_data', required=False, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    #parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--batch_size', type=int, default=300, help='input batch size')
    # parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    # parser.add_argument('--num_iter', type=int, default=6000, help='number of iterations to train for')
    parser.add_argument('--num_iter', type=int, default=30000, help='number of iterations to train for')
    #parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    #parser.add_argument('--valInterval', type=int, default=1000, help='Interval between each validation')
    parser.add_argument('--valInterval', type=int, default=1000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    # parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
    #                     help='assign ratio for each selected data in the batch')
    parser.add_argument('--batch_ratio', default='1', help='assign ratio for each selected data in the batch')

    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    #parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    #parser.add_argument('--batch_max_length', type=int, default=200, help='maximum-label-length')
    parser.add_argument('--batch_max_length', type=int, default=300, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    # parser.add_argument('--character', type=str,
    #                     default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')

    # 원하는 문자열을 utf-8로 인코딩한 다음 다시 utf-8로 디코딩하여 문자열을 설정
    #opt_character = ' ()+-./01234569ACDFGILQSTWabeghimnoprstuz~°กขคงจฉชซฑณดตถทธนบปผฝพฟภมยรลวศษสหฬอะัาำิีึืุูเแโใไ็็่้๊์✓'
    opt_character = ' ()+-./01234569ACDFGILQSTWabeghimnoprstuz~°กขคงจฉชซฑณดตถทธนบปผฝพฟภมยรลวศษสหฬอะัาำิีึืุูเแโใไๆ็่้๊์✓'

    opt_character = '()+-./01234569ACDFGILQSTWabeghimnoprstuz~°กขคงจฉชซฑณดตถทธนบปผฝพฟภมยรลวศษสหฬอะัาำิีึืุูเแโใไๆ็่้๊์✓'.encode(
        'utf-8').decode('utf-8')

    # 두 문자열을 합침
    combined_text = "¢£¤¥!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZกขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤเแโใไะาุูิีืึั่้๊๋็์ำํฺฯๆ0123456789๑๒๓๔๕๖๗๘๙" + \
                    "()+-./01234569ACDFGILQSTWabeghimnoprstuz~°กขคงจฉชซฑณดตถทธนบปผฝพฟภมยรลวศษสหฬอะัาำิีึืุูเแโใไๆ็่้๊์✓"

    # 중복 제거
    unique_combined_text = ''.join(sorted(set(combined_text), key=combined_text.index))
    # ¢£¤¥!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZกขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤเแโใไะาุูิีืึั่้๊๋็์ำํฺฯๆ0123456789๑๒๓๔๕๖๗๘๙°✓
    # 결과 출력
    print(unique_combined_text)

    # parser.add_argument('--character', type=str,
    #                     default=' ()+-./01234569ACDFGILQSTWabeghimnoprstuz~°กขคงจฉชซฑณดตถทธนบปผฝพฟภมยรลวศษสหฬอะัาำิีึืุูเแโใไๆ็่้๊์✓',
    #                     help='character label')
    # 기존 오픈소스 모델 character
    # parser.add_argument('--character', type=str,
    #                     default='¢£¤¥!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZกขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤเแโใไะาุูิีืึั่้๊๋็์ำํฺฯๆ0123456789๑๒๓๔๕๖๗๘๙',
    #                     help='character label')
    parser.add_argument('--character', type=str,
                        default=unique_combined_text,
                        help='character label')
    # 원본 캐릭터
    #'¢£¤¥!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZกขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤเแโใไะาุูิีืึั่้๊๋็์ำํฺฯๆ0123456789๑๒๓๔๕๖๗๘๙'
    # parser.add_argument('--character', type=str,
    #                     default=opt_character, help='character label')

    #" ()+-./01234569ACDFGILQSTWabeghimnoprstuz~°กขคงจฉชซฑณดตถทธนบปผฝพฟภมยรลวศษสหฬอะัาำิีึืุูเแโใไๆ็่้๊์✓"
    #parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--sensitive', action='store_true', default=True, help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    # parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--data_filtering_off', action='store_true', default=True, help='for data_filtering_off mode')

    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=False, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=False,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=False, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=False, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.character : {opt.character}')

    opt = parser.parse_args()
    #data_lmdb_release\training\ttf --valid_data data_lmdb_release\validation\ttf --select_data th --batch_ratio 1 --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC

    # 필수 옵션 지정
    #opt.valid_data = "data_lmdb_release/validation/ttf"
    #opt.train_data = "data_lmdb_release/training/ttf"
    opt.valid_data = "data_lmdb_release/validation/ttf3"
    opt.train_data = "data_lmdb_release/training/ttf3"
    opt.Transformation = "None"
    opt.FeatureExtraction = "VGG"
    opt.SequenceModeling = "BiLSTM"
    opt.Prediction = "CTC"
    opt.select_data = "th"
    opt.batch_ratio = "1"

    print(opt.valid_data)
    print(opt.train_data)
    # if 'ๆ' in opt.character:
    #     # 'ๆ' 문자가 있을 경우
    #     #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!야가 문제야???? 막판에 내가 그냥 하드코딩함 미세하게 아래로 안가면 있음')
    # else:
    #     # 'ๆ' 문자가 없을 경우
    #     #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!야가 문제야???? 막판에 내가 그냥 하드코딩함 미세하게 합쳐지면 없음')

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt = parser.parse_args() 후에 갑자기 바뀌어 버림  opt.character : {opt.character}')

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)


    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!센서티브에서 지맘대로 아스키로 바꾸기전  !!!!!!!!!! opt.character : {opt.character}')
        print(
            f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!센서티브 string.printable : {string.printable}')
        #opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!센서티브에서 지맘대로 아스키로 바꾼후!!!!!!(use 94 char) 어쩌고 저쩌고   !!!!!!!!!! opt.character : {opt.character}')


    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)

    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈 업데이트 했음 ㅜ 변경전 opt.num_gpu : {opt.num_gpu}')



    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1

        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈 업데이트 했음 ㅜ 변경전 opt.workers : {opt.workers}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈 업데이트 했음 ㅜ 변경전 opt.batch_size : {opt.batch_size}')
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu


        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.num_gpu : {opt.num_gpu}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈 업데이트 했음 ㅜ 변경후 opt.workers : {opt.workers}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈 업데이트 했음 ㅜ 변경후 opt.batch_size : {opt.batch_size}')


        opt.batch_size = 300
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈 업데이트 했음 ㅜ 300 변경후 opt.batch_size : {opt.batch_size}')
        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """
    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!메인함수에서 트레인 함수 에 opt 전달할 때 태국어가 없어짐 아규먼트 함수때문에 그럼??? !!!!!!!!!! opt.character : {opt.character}')
    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 에 마지막 넘겨주기 전 여기서 디코드 하기 전엔 정상임 opt.character : {opt.character}')
    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 에 마지막 넘겨주기 전전전 여기서 디코드해도 정상으로 감 opt.character : {opt.character}')
    opt_character = opt.character.encode('utf-8').decode('utf-8')
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train 에 마지막 넘겨주기 전 여기서 디코드해도 정상으로 감 opt.character : {opt.character}')
    # if 'ๆ' in opt.character:
    #     # 'ๆ' 문자가 있을 경우
    #     #print('train 에 마지막 넘겨주기 전 여기서 디코드해도 정상으로 감  있음')
    # else:
    #     # 'ๆ' 문자가 없을 경우
    #     print('train 에 마지막 넘겨주기 전 여기서 디코드해도 정상으로 감  없음')

    train(opt)
