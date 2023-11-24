import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import datetime

get_batch_cnt = 0
class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """

        current_date = datetime.datetime.now().strftime("%m-%d")
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!dataset 파일 current_date {current_date}')

        # 원하는 폴더 경로 생성
        directory = f'./saved_models/{opt.exp_name}/{current_date}/'

        # 폴더가 존재하지 않으면 생성
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 파일 열기
        #log = open(f'{directory}/log_dataset.txt', 'a')
        log = open(f'{directory}/log_dataset.txt', 'a', encoding='utf-8')

        #log = open(f'./saved_models/{opt.exp_name}/{current_date}/log_dataset.txt', 'a')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!dataset 파일 log : {log}')

        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Batch_Balanced_Dataset  이닛 opt  : {opt}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Batch_Balanced_Dataset  이닛 opt.character : {opt.character}')
        if 'ๆ' in opt.character:
            # 'ๆ' 문자가 있을 경우
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Batch_Balanced_Dataset  이닛 opt.character 있음')
        else:
            # 'ๆ' 문자가 없을 경우
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Batch_Balanced_Dataset  이닛 opt.character 없음')

        print(
            f'Batch_Balanced_Dataset dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        print(
            f'Batch_Balanced_Dataset dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(
            f'Batch_Balanced_Dataset dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0

        print(f'@@@@@@@@@@Batch_Balanced_Dataset 태국어도 학습하라고 ㅜㅜㅜㅜㅜㅜ   opt.select_data : {opt.select_data}')

        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            print(f'Batch_Balanced_Dataset for 루프 2번 들어오라고 ㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜ@@@@@@@@@@태국어도 학습하라고 ㅜㅜㅜㅜㅜㅜ   selected_d : {selected_d}')
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈가 각 데이터 디렉토리마다 다르게 계산됨 opt.batch_ratio : {opt.batch_ratio}')

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈가 각 데이터 디렉토리마다 다르게 계산됨 opt.select_data : {opt.select_data}')

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!여기서 배치사이즈가 각 데이터 디렉토리마다 다르게 계산됨 batch_size : {_batch_size}')


            print(dashed_line)
            log.write(dashed_line + '\n')
            # print("읽히라고!!!!!!!!!! 개짜증나 ㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜㅜ")
            print(f'Batch_Balanced_Dataset opt.train_data : {opt.train_data}')

            if 'ๆ' in opt.character:
                # 'ๆ' 문자가 있을 경우
                print('배치 밸런스 넘어오면서 없어지는 거임? 있음')
            else:
                # 'ๆ' 문자가 없을 경우
                print('배치 밸런스 넘어오면서 없어지는 거임? 없음')

            print(f'Batch_Balanced_Dataset [selected_d] : {[selected_d]}')

            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            print(f'Batch_Balanced_Dataset _dataset : {_dataset}')
            print(f'Batch_Balanced_Dataset _dataset_log : {_dataset_log}')
            print(f'Batch_Balanced_Dataset opt.train_data : {opt.train_data}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Batch_Balanced_Dataset 이닛 opt : {opt}')
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!_dataset_log : {_dataset_log}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!total_number_dataset : {total_number_dataset}')
            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!total_number_dataset : {total_number_dataset}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.total_data_usage_ratio : {opt.total_data_usage_ratio}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!number_dataset : {number_dataset}')

            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!indices  range(total_number_dataset) : {indices}')

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!dataset_split : {dataset_split}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!zip(_accumulate(dataset_split), dataset_split) : {zip(_accumulate(dataset_split), dataset_split)}')
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]

            print(f"!!!!!!!!!!!!!!!!!컴프리헨션 분해 하기 전 _dataset: {_dataset} ")


            _dataset, a = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            print(f"!!!!!!!!!!!!!!!!!컴프리헨션 분해 하기 전 _dataset: {_dataset} ")
            print(f"!!!!!!!!!!!!!!!!!컴프리헨션 분해 하기 전 a: {a} ")


            for offset, length in zip(_accumulate(dataset_split), dataset_split):
                print(f"!!!!!!!!!!!!!!!!!컴프리헨션 분해 Offset: {offset}, Length: {length}")
                subset = Subset(_dataset, indices[offset - length:offset])
                print(f"!!!!!!!!!!!!!!!!!컴프리헨션 분해 subset: {subset} ")
                # 여기에서 subset을 사용하여 작업 수행



            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!_batch_size : {_batch_size}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!batch_size_list : {batch_size_list}')
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Total_batch_size : {Total_batch_size}')

            # print( '@@@@@@@@@@@이제 데이터 로더가 안돼 ㅜㅜㅜㅜㅜ 개짜증나 ㅜㅜㅜㅜㅜㅜㅜㅜㅜ')
            # print(f'@@@@@@@@@@@@@@@@@이제 데이터 로더가 안돼 opt.workers : {opt.workers}')

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
            print(f'%%%%%%%%%%%%%%%%%% 여기가 문제 len dataloader_iter_list : {len(self.dataloader_iter_list)}')

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!batch_size_list : {batch_size_list}')


        print(Total_batch_size_log)
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!batch_size_sum : {batch_size_sum}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!opt.batch_size : {opt.batch_size}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Total_batch_size : {Total_batch_size}')

        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Total_batch_size_log : {Total_batch_size_log}')
        log.write(Total_batch_size_log + '\n')
        log.close()

    #balanced_batch_texts = []

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []
        dataloader_iter_list_cnt = 0

        # print(f'$$$$$$$$$$$$$$$$$$$$$ dataloader_iter_list 길이 : {len(dataloader_iter_list)}')
        #get_batch_cnt = get_batch_cnt +1
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!get_batch_cnt : {get_batch_cnt}')
        #print(f'get_batch self.dataloader_iter_list : {self.dataloader_iter_list}')
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            print(f'get_batch i : {i}')
            dataloader_iter_list_cnt = dataloader_iter_list_cnt + 1
            print(f'get_batch for 반복 dataloader_iter_list dataloader_iter_list_cnt : {dataloader_iter_list_cnt}')
            try:
                image, text = next(data_loader_iter)
                #print(f'get_batch  next(data_loader_iter) text : {text}')
                balanced_batch_images.append(image)
                balanced_batch_texts += text
                #print(f'get_batch  enumerate next(data_loader_iter) balanced_batch_texts : {balanced_batch_texts}')
                print(f'get_batch  enumerate next(data_loader_iter) balanced_batch_texts len: {len(balanced_batch_texts)}')


            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass
        print(f'get_batch dataloader_iter_list 반복 완료 get_batch_cnt : {get_batch_cnt}')

        #print(f'get_batch dataloader_iter_list 반복 완료 get_batch_cnt : {get_batch_cnt}')

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        #print(f'!!!!!!!!!!!!!!!get_batch 반환전 balanced_batch_images : {balanced_batch_images}')
        print(f'!!!!!!!!!!!!!!!get_batch 반환전 길이 balanced_batch_images : {len(balanced_batch_images)}')
        #print(f'!!!!!!!!!!!!!!!get_batch 반환전 balanced_batch_texts : {balanced_batch_texts}')
        print(f'!!!!!!!!!!!!!!!get_batch 반환전 길이 balanced_batch_texts : {len(balanced_batch_texts)}')
        return balanced_batch_images, balanced_batch_texts


# def hierarchical_dataset(root, opt, select_data='/'):
def hierarchical_dataset(root, opt, select_data=''):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []

    # print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ root : {root}')
    # print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ select_data : {select_data}')
    # print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ select_data[0] : {select_data[0]}')

    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    print(f'hierarchical_dataset hierarchical_dataset root : {root}')
    print(f'hierarchical_dataset select_data : {select_data}')

    dataset_log += '\n'
    # for dirpath, dirnames, filenames in os.walk(root+'/'):
    # for dirpath, dirnames, filenames in os.walk(root + '\\'):
    for dirpath, dirnames, filenames in os.walk(root + r''):

        print(f'hierarchical_dataset hierarchical_dataset dirpath : {dirpath}')
        print(f'hierarchical_dataset hierarchical_dataset dirnames : {dirnames}')
        print(f'hierarchical_dataset hierarchical_dataset filenames : {filenames}')

        if not dirnames:
            select_flag = False
            for selected_d in select_data:

                print(f'있다 dirpath : {dirpath}')
                #if selected_d in dirpath:
                if selected_d in dirpath and filenames:
                    print(f'있다 dirpath : {dirpath}')
                    print(f'디렉토리 {dirpath}에 파일이 있음')
                    print(f'있다 selected_d : {selected_d}')
                    print(f'있다 dirpath : {dirpath}')
                    select_flag = True
                    break

            if select_flag:
                # print(f'여기서 이상하게 보냄 dirpath : {dirpath}')
                # print(f'여기서 이상하게 보냄 opt : {opt}')
                dataset = LmdbDataset(dirpath, opt)

                # 라벨 정보 가져오기
                labels2 = []

                for i in range(len(dataset)):
                    key = str(i)  # 각 데이터 포인트의 키를 순차적으로 생성
                    #value = dataset.get_item(key)  # 키에 해당하는 값(라벨 및 이미지 데이터) 가져오기
                    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hierarchical_dataset 라벨 정보 가져오기 key : {key}')
                    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hierarchical_dataset 라벨 정보 가져오기 value : {value}')
                    #label = value['label']  # 'label' 키를 사용하여 라벨 정보 가져오기
                    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hierarchical_dataset 라벨 정보 가져오기 label : {label}')
                    #labels2.append(label)
                #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hierarchical_dataset 라벨 정보 가져오기 labels2 : {labels2}')
                # 라벨 출력
                #print("Labels from LmdbDataset labels2:", labels2)

                # 중복된 라벨 확인
                #unique_labels = set(labels2)
                #ㅃprint("!!!!!!!!!!!!!!!여기서 중복된거 제거돼서 71 개임/??Number of Unique Labels:", len(unique_labels))

                # print(f' dirpath : {dirpath}')
                # print(f'  root : {root}')
                # print(f' os.path.relpath(dirpath, root) : {os.path.relpath(dirpath, root)}')

                # print(f' len(dataset) : {len(dataset)}')

                relative_path = os.path.relpath(dirpath, root)
                relative_path = relative_path.replace('/', '\\')  # 슬래시를 역슬래시로 변경

                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hierarchical_dataset len(dataset) : {len(dataset)}')
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hierarchical_dataset dataset : {dataset}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hierarchical_dataset dataset_list : {dataset_list}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!hierarchical_dataset dataset_log : {dataset_log}')

    concatenated_dataset = ConcatDataset(dataset_list)
    #print(f'hierarchical_dataset concatenated_dataset : {concatenated_dataset}')
    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt

        print(f'LmdbDataset 여기--------------------root {root}')

        print(f'LmdbDataset opt : {opt}')
        if 'ๆ' in opt.character:
            # 'ๆ' 문자가 있을 경우
            print('LmdbDataset 있음')
        else:
            # 'ๆ' 문자가 없을 경우
            print('LmdbDataset 없음')

        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!LmdbDataset nSamples : {nSamples}')
            print(f'###########################################LmdbDataset self.nSamples  : {self.nSamples}')

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!LmdbDataset self.filtered_index_list : {self.filtered_index_list}')
                print(
                    f'^^^^^^^^^^^^^^^^^OKOKOK 필터링 하지마!!!!!!!!!!!!!!!!!!self.filtered_index_list : {self.filtered_index_list}')
                print(
                    f'^^^^^^^^^^^^^^^^^필터링 하지마!!!!!!!!!!!!!!!!!!self.nSamples : {self.nSamples}')
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!dataset.py __init__  label_key : {label_key}')
                    label = txn.get(label_key).decode('utf-8')
                    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!dataset.py __init__  label : {label}')

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^설마 여기서 필터링 됨?????')
                        continue

                    self.filtered_index_list.append(index)
                    print(f'self.filtered_index_list : {self.filtered_index_list}')
                    print(f'len(self.filtered_index_list) : {len(self.filtered_index_list)}')
                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            # label = re.sub(out_of_char, '', label)
            # 이미지를 시각화하고 저장
            #img.show()  # 이미지를 화면에 표시
        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
