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
import lmdb
import os
from PIL import Image

class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        print(f'!!!!!!!!!!!!!!!!!Batch_Balanced_Dataset 함수 호출함!!!!!!!!!!!!!!!! opt : {opt}')
        print('?????????????????????')
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print('?????????????????????')

        opt.select_data = [
            'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata'
            , 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/validation/thdata'
        ]

        # opt.select_data = [
        #     'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/data.mdb'
        #     , 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/data.mdb'
        # ]

        print('바꿈')
        opt.batch_ratio = ['0.5',
                           '0.5']
        #opt.batch_ratio = 0.5
        print('Batch_Balanced_Dataset batch_ratio 바꿈')
        print(f'Batch_Balanced_Dataset train_data: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'Batch_Balanced_Dataset dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        print('짜증나___________________________어떻게 하라고 ㅜㅜㅜㅜㅜㅜㅜㅜ')
        print(f'len(opt.select_data): {len(opt.select_data)}')
        print(f'len(opt.batch_ratio): {len(opt.batch_ratio)}')
        print(f'52 트레인 데이터가 틀림???????     opt.train_data: {opt.train_data}')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            print('여기서 에러남????????')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            print('selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio)')
            print(f'_dataset: {_dataset}')

            total_number_dataset = len(_dataset)
            print(f'opt.train_data: {opt.train_data}')
            print(f'0 나오면 안 되는데 0 나옴+_========= : {_dataset}  {len(_dataset)}')
            print(f'total_number_dataset: {total_number_dataset}')
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
        print(f'_data_loader : {_data_loader}')

        print(f'이건 또 뭐야  데이터 로더  Batch_Balanced_Dataset 이건 또 뭐야  데이터 로더   len(data_loader_list) : {len(data_loader_list)}')

        print(f'이건 또 뭐야  데이터 로더  Batch_Balanced_Dataset len(dataloader_iter_list) : {len(dataloader_iter_list)}')


        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    print('hierarchical_dataset!!!!!!!!!!!!!!!')
    dataset_log = f'hierarchical_dataset 계층 루트????   dataset_root: ????????????   {root}\t dataset: {select_data[0]}'
    print('dataset_log!!!!!!!!!!!!!!!')
    print(dataset_log)
    dataset_log += '\n'

    root = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/data.mdb'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        print(f'os.walk(root root : {root}')
        print(f'os.walk(root dirpath : {dirpath}')
        print(f'os.walk(root dirnames : {dirnames}')
        print(f'os.walk(root filenames : {filenames}')
        print('====================================================================')
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                print(f'~~~~~~~~~~~~~~~~~for selected_d in select_data:  selected_d : {selected_d}')
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                #dirpath = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata'
                #dirpath = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/data.mdb'
                #dirpath = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/data.mdb/data.mdb'
                print(f'111111((((((((((@@@@@@@@@@@@@################################hierarchical_dataset 여기서 리턴할때 dataset 이 없음 dirpath : {dirpath}')
                print(f'22222((((((((((((((((@@@@@@@@@@@################################hierarchical_dataset 여기서 리턴할때 dataset 이 없음 root : {root}')
                dataset = LmdbDataset(dirpath, opt)
                print(f'33333(((((((((((((((LmdbDataset 실행함!!!!!!!!!! len(dataset) : {len(dataset)}')
                print(f'(((((((((((((((@@@@@@@@@@################################hierarchical_dataset 여기서 리턴할때 dataset 이 없음 len(dataset) : {len(dataset)}')
                print(f'163  ~~~~~~~~~~ 여기서 리턴 데이터 셋 dirpath : {dirpath}')
                print(f'~~~~~~~~~~ 여기서 리턴 데이터 셋 len(dataset) : {len(dataset)}')
                print('145  ???????????개어려워ㅏ')
                print(f'dirpath : {dirpath} \t ====================  opt : {opt}')
                print('opt 개길어========================================')
                print(f'dirpath : {root} \t ============  opt : {root}')
                print(f'148 dataset : {dataset} \t  ')
                print(f'157  @@@@@@@@@@@@@@@@@@@@@@@@@149 len(dataset) : {len(dataset)} \t  ')
                dataset_log = f'dirpath:    {dirpath}\t root: {root}os.path.relpath:    {os.path.relpath(dirpath, root)}\t len(dataset): {len(dataset)}'
                print("개짜증나 진짜 ㅡ+++++++++++++++++++++++++++++++")
                #import lmdb

                # 데이터베이스 경로 설정
                #dirpath = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/output.lmdb'
                dirpath = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/data.mdb'
                # LMDB 데이터베이스 열기
                env = lmdb.open(dirpath, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

                # 데이터베이스 내 데이터 수 세기
                def count_data(env):
                    with env.begin() as txn:
                        #nSamples = int(txn.get("num-samples".encode()))

                        num_samples_key = 'num-samples'.encode()
                        num_samples_value = txn.get(num_samples_key)
                        print(f'!@@@@@@@@@@@@@@@txn.get(num_samples_key) num_samples_value : {num_samples_value}')
                        if num_samples_value is not None:
                            num_samples = int(num_samples_value)
                            print(f"이미지 샘플의 총 수: {num_samples}")
                        else:
                            print("이미지 샘플의 총 수를 찾을 수 없습니다.")

                        # nSamples = int(txn.get('num-samples'.encode()))
                        nSamples = num_samples


                    return nSamples

                num_samples = count_data(env)

                print(f"Number of samples in the database: {num_samples}")



                sub_dataset_log = f'193 죽겠네 진짜 ㅜㅜㅜㅜ sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)
                print(f' @@@@@@@@@@@@@@@@@@@ 여기서 리턴 len(dataset) : {len(dataset)}')
                print(f' @@@@@@@@@@@@@@@@@@@ 여기서 리턴 dataset_list : {dataset_list}')
                print(f' @@@@@@@@@@@@@@@@@@@ 여기서 리턴 len(dataset_list) : {len(dataset_list)}')
    concatenated_dataset = ConcatDataset(dataset_list)
    print(f' @@@@@@@@@@@@@@@@@@@ 여기서 리턴 concatenated_dataset : {concatenated_dataset}')
    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):
    print(f'44444444((((((((((((((((LmdbDataset 클래스 들어옴!!!!!!!!!!!!!!!! Dataset : {Dataset}')
    def __init__(self, root, opt):
        #import lmdb

        # 이후 코드를 계속 진행합니다.
        print(f'44444444((((((((((((((((LmdbDataset 클래스 들어옴 __init__ 함수 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 처음 들어올때 opt opt : {opt}')

        print(f'44444444((((((((((((((((LmdbDataset 클래스 들어옴 __init__ 함수 LmdbDataset 함수 lmdb 초기화 해야 함')
        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            print('44444444((((((((((((((((LmdbDataset 클래스 들어옴 진짜 어렵다 ㅜㅜㅜㅜㅜㅜㅜ')
            print(f'44444444((((((((((((((((LmdbDataset 클래스 들어옴 LmdbDataset root 경로 : {self.root}')


            # 데이터 경로
            data_path = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata'

            # LMDB 데이터베이스 경로
            lmdb_path = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/data.mdb'

            # 디렉토리 생성
            lmdb_dir = os.path.dirname(lmdb_path)
            os.makedirs(lmdb_dir, exist_ok=True)

            #import lmdb
            # for root, dirs, files in os.walk(data_path):
            #     for filename in files:
            #         if filename.endswith(('.jpg', '.jpeg', '.png')):
            #             img_path = os.path.join(root, filename)
            #             img = Image.open(img_path)
            #             img_data = img.tobytes()
            #             # 이미지 파일 이름을 키로 사용
            #             key = os.path.basename(img_path).encode('utf-8')
            #             # 이미지 파일명을 라벨로 사용
            #             label = os.path.basename(img_path).encode('utf-8')
            #             txn.put(key, img_data)
            #             txn.put(label, label)  # 이미지 파일명을 라벨로 저장
            #
            #             # "num-samples" 키를 추가하여 이미지 샘플 수 설정
            #             print("짜증나++++++++++++++++++++++++++++++")
            #             txn.put('num-samples'.encode(), str(len(image_paths)).encode())

            # LMDB 데이터베이스 경로
            lmdb_path = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata'
            lmdb_path = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/data.mdb'
            # LMDB 환경 열기 (readonly 모드로 열기)
            env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

            # 데이터베이스에서 'num-samples' 키에 해당하는 값을 가져와서 출력
            try:
                with env.begin(write=False) as txn:

                    cursor = txn.cursor()
                    for key, _ in cursor:
                        print(f"###################Key: {key.decode('utf-8')}")


                    num_samples_key = 'num-samples'.encode()
                    num_samples_value = txn.get(num_samples_key)
                    if num_samples_value is not None:
                        num_samples = int(num_samples_value)
                        print(f"이미지 샘플의 총 수: {num_samples}")
                    else:
                        print("이미지 샘플의 총 수를 찾을 수 없습니다.")


                    #nSamples = int(txn.get('num-samples'.encode()))
                    nSamples = num_samples
                    print(f'nSamples ???? : {nSamples}')
                    self.nSamples = nSamples

                    self.filtered_index_list = []
                    for index in range(self.nSamples):
                        #index += 1  # lmdb starts with 1
                        index += 0  # lmdb starts with 1
                        print(f"Current index: {index}")  # 추가

                        #label_key = 'label-%09d'.encode() % index
                        label_key = 'label-C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/image_%01d.png'.encode() % index
                        #label_key = 'label-image_0.png'.encode() % index
                        print(f"label_key: {label_key}")  # 추가
                        label = txn.get(label_key).decode('utf-8')
                        # 나머지 코드...



                    if self.opt.data_filtering_off:
                        # for fast check or benchmark evaluation with no filtering
                        self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
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

                            #index += 1  # lmdb starts with 1
                            index += 0  # lmdb starts with 1
                            #label_key = 'label-%09d'.encode() % index
                            label_key = 'label-C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/data/train-easyocr/step3/training/thdata/image_%01d.png'.encode() % index
                            print(f'308 번 줄 label_key : {label_key}')
                            print(f'309 번 줄 txn.get(label_key) : {txn.get(label_key)}')
                            label = txn.get(label_key).decode('utf-8')

                            if len(label) > self.opt.batch_max_length:
                                # print(f'The length of the label is longer than max_length: length
                                # {len(label)}, {label} in dataset {self.root}')
                                continue

                            # By default, images containing characters which are not in opt.character are filtered.
                            # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                            out_of_char = f'[^{self.opt.character}]'
                            if re.search(out_of_char, label.lower()):
                                continue

                            self.filtered_index_list.append(index)

                        self.nSamples = len(self.filtered_index_list)
            # 환경 닫기
            finally:
                env.close()

    def __len__(self):
        print(f'%%%%%%%%%%!!!!!!!%%%%%%%%%%%%%%%%%%__len__self.nSamples  : {self.nSamples}')
        return self.nSamples

    def __getitem__(self, index):
        print(f'%%%%%%%%%%!!!!!!!%%%%%%%%%%%%%%%%%%__getitem__index__getitem__index__getitem__index__getitem__index__getitem__index__getitem__index  : {index}')
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
            label = re.sub(out_of_char, '', label)
        print(f'++++++++++++++++++++++++44444444((((((((((((((((__getitem__ 언제 호출함????? LmdbDataset 클래스 들어옴!!!!!!!!!!!!!!!! img : {img}')
        print(f'+++++++++++++++++++++++44444444(((((((((((((((__getitem__ 언제 호출함?????(label 클래스 들어옴!!!!!!!!!!!!!!!! label : {label}')
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
