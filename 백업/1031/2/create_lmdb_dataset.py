""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


import os
print('create_lmdb_datset')

# 환경 변수 이름
env_variable_name = "inputPath"

# 환경 변수 읽어오기
env_variable_value = os.getenv(env_variable_name)
# 읽어온 환경 변수 출력
print(f"환경 변수 {env_variable_name}의 값: {env_variable_value}")

# 환경 변수 이름
env_variable_name = "outputPath"

# 환경 변수 읽어오기
env_variable_value = os.getenv(env_variable_name)
# 읽어온 환경 변수 출력
print(f"환경 변수 {env_variable_name}의 값: {env_variable_value}")

# 환경 변수 이름
env_variable_name = "gtFile"

# 환경 변수 읽어오기
env_variable_value = os.getenv(env_variable_name)

# import os
#
# # 모든 환경 변수 출력
# for key, value in os.environ.items():
#     print(f"{key}: {value}")


# 읽어온 환경 변수 출력
print(f"환경 변수 {env_variable_name}의 값: {env_variable_value}")



def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    print(f'outputPath : {outputPath}')
    print(f"createDataset 들어옴 outputPath  Opening lmdb at path: {outputPath}")
    #env = lmdb.open(outputPath, map_size=1099511627776)
    env = lmdb.open(outputPath, map_size=109951 )
    cache = {}
    cnt = 1
    print('createDataset 들어옴___________')
    print(f'inputPath : {inputPath}')

    print(f'gtFile : {gtFile}')

    print(f'outputPath : {outputPath}')


    with open(gtFile, 'r', encoding='utf-8') as data:

        datalist = data.readlines()
        print(f'with open(gtFile  datalist : {datalist}')


    nSamples = len(datalist)
    for i in range(nSamples):
        print(f'변경전 datalist[i] : {datalist[i]}')
        imagePath, label = datalist[i].strip('\n').split('\t')

        print(f'변경전 inputPath : {inputPath}')
        print(f'변경전 imagePath : {imagePath}')
        imagePath = os.path.join(inputPath, imagePath)
        print(f'변경후 imagePath : {imagePath}')
        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        print(f'imageKey : {imageKey}')
        print(f'labelKey : {labelKey}')

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    print(f'__main__ 들어옴: ')

    inputPath = 'data_lmdb_release/training/JH/JH_train'  # 필수 입력 인수 설정
    gtFile = 'data_lmdb_release/training/JH/gt/gt.txt'  # 필수 입력 인수 설정
    outputPath = 'data_lmdb_release/training/JH/mdb'  # 필수 입력 인수 설정

    print(inputPath)
    print(gtFile)
    print(outputPath)

    fire.Fire(createDataset)
