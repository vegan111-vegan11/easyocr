import torch
from pythainlp import word_tokenize, sent_tokenize
#from pythainlp.util import thai_str2int

# def thai_text_to_int(text):
#     # 태국어 문자열을 정수로 변환
#     int_list = thai_str2int(text)
#     return int_list


# 태국어 문자에서 정수로의 간단한 매핑 예시
char_to_int = {
    'ล': 1,
    'ด': 2,
    'ส': 3,

'า': 4,
    'ร': 5,
    'ก่': 6,

'อ': 7,
    'ภู': 8,
    'มิ': 9,


'แ': 10,
    'พ้': 11,

't': 12,
    'e': 13,
    's': 14,
'1': 15,
'2': 16,
' ': 17,


'ก': 18,
'่': 19,
'ภ': 20,

 'ู': 21,
'ม': 22,
 'ิ': 23,

#'แ': 24,
'พ': 24,
 '้': 25






    # 나머지 문자에 대한 매핑 계속
}

def thai_text_to_int(text):
    print(f'thai_text_to_int 함수 들어옴  text : {text}')

    int_list = [char_to_int[char] for char in text if char in char_to_int]
    print(f'thai_text_to_int 리턴 int_list : {int_list}')
    print('********************************')
    return int_list


text = "สวัสดีครับ ทุกคน"
text = "ลดสารก่อภูมิแพ้"

# 단어 토큰화
words = word_tokenize(text)
print(words)  # ['สวัสดี', 'ครับ', 'ทุกคน']
print(f'words : {words}')

# 문장 토큰화
sentences = sent_tokenize(text)
print(sentences)  # ['สวัสดีครับ', 'ทุกคน']
print(f'sentences : {sentences}')
print('******************************')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        print(f'=============== CTCLabelConverter 왜 첨자가 따로 인식돼지????  character: {character}')

        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

            print(f'==============CTCLabelConverter i : {i}')

            print(f'=============== CTCLabelConverter : {char}')


        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            # text = [self.dict[char] for char in text]

            print(f'태국어 문자열 text : {text}')
            print(f'thai_text_to_int 함수 들어가라니까  태국어 문자열 t : {t}')
            print(f'thai_text_to_int 함수 들어가라니까  태국어 문자열 text : {text}')

            int_list = thai_text_to_int(text)  # 문자열을 정수 목록으로 변환

            print(f'thai_text_to_int 함수 들어가라니까  태국어 문자열 int_list : {int_list}')
            #tensor = torch.LongTensor(int_list)
            batch_text[i][:len(text)] = torch.LongTensor(text)
            #batch_text[i][:len(text)] = torch.LongTensor(int_list)

            #batch_text[i][:len(text)] = torch.LongTensor(text)

        return (batch_text.to(device), torch.IntTensor(length).to(device))


def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.

        print(f'@@@@@@@@CTCLabelConverterForBaiduWarpctc character : {character}')


        dict_character = list(character)

        print(f'@@@@@@@@CTCLabelConverterForBaiduWarpctc dict_character : {dict_character}')


        self.dict = {}
        for i, char in enumerate(dict_character):
            print(f'======================@@@@@@@@CTCLabelConverterForBaiduWarpctc for i, char in enumerate(dict_character) i : {i}')

            print(f'======================@@@@@@@@CTCLabelConverterForBaiduWarpctc for i, char in enumerate(dict_character) char : {char}')

            print(f'======================@@@@@@@@CTCLabelConverterForBaiduWarpctc for i, char in enumerate(dict_character) char : {dict_character}')




            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """


    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

            print(f'==============AttnLabelConverter i : {i}')

            print(f'=============== AttnLabelConverterchar : {char}')

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        #################
        print(f'~~~~~~~~~~~~~~length : {length}')


        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        #batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)

        print(f'~~~~~~~~~~~~~~batch_text : {batch_text}')
        for i, t in enumerate(text):
            # text = list(t)
            # text.append('[s]')
            # text = [self.dict[char] for char in text]
            # batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token

            print(f'AttnLabelConverter batch_text.to(device) : {batch_text.to(device)}')
            print(f'AttnLabelConverter torch.IntTensor(length).to(device) : {torch.IntTensor(length).to(device)}')
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
