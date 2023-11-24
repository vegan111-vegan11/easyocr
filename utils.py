import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py device : {device}')

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.

        # print(f'CTCLabelConverter 들어옴 이닛함수 처음 character 들어왔을때는 있음???    여기서 ๆ 가 있음????? !!!!!!!!!!!!!!character : {character}')
        # print(
        #     f'CTCLabelConverter 들어옴 이닛함수 처음 character 들어왔을때는 있음???    여기서 ๆ 가 있음????? !!!!!!!!!!!!!!len(character) : {len(character)}')
        # if '✓' in character:
        #     # 'ๆ' 문자가 있을 경우
        #     print('CTCLabelConverter 들어옴 이닛함수 처음 character 들어왔을때 있음')
        # else:
        #     # 'ๆ' 문자가 없을 경우
        #     print('CTCLabelConverter 들어옴 이닛함수 처음 character 들어왔을때 없음')

        dict_character = list(character)

        # print(f'CTCLabelConverter 들어옴 이닛함수 여기서 ๆ 가 없어짐 list(character) 로 바꿔서??????? !!!!!!!!!!!!!!character : {character}')
        # print(
        #     f'CTCLabelConverter 들어옴 이닛함수 여기서 ๆ 가 없어짐 list(character) 로 바꿔서??????? !!!!!!!!!!!!!!dict_character = list(character) 후 dict_character : {dict_character}')

        # if 'ๆ' in character:
        #     # 'ๆ' 문자가 있을 경우
        #     print('CTCLabelConverter 들어옴 이닛함수 여기서 ๆ 가 없어짐 list(character) 로 바꿔서 있음')
        # else:
        #     # 'ๆ' 문자가 없을 경우
        #     print('CTCLabelConverter 들어옴 이닛함수 여기서 ๆ 가 없어짐 list(character) 로 바꿔서 없음')

        self.dict = {}
        #print(f'!!!!!!!!!!!!CTCLabelConverter 이닛함수 들어옴 여기서 ๆ 가 없음 유티에프8로 인코딩 해버려서 이상하게 들어왔음 dict_character : {dict_character}')



        for i, char in enumerate(dict_character):
            # print(
            #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py __init__  for i, char in enumerate(dict_character) dict_character : {dict_character}')
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            # print(
            #     f'CTCLabelConverter 들어옴 이닛함수 for i, char in enumerate(dict_character) i #### ??????? !!!!!!!!!!!!!!character i : {i}')
            # print(
            #     f'CTCLabelConverter 들어옴 이닛함수 for i, char in enumerate(dict_character) i #### ??????? !!!!!!!!!!!!!!i + 1 : {i + 1}')
            # print(
            #     f'CTCLabelConverter 들어옴 이닛함수 여기서 for i, char in enumerate(dict_character) char ### ??????? !!!!!!!!!!!!!!character : {char}')
            self.dict[char] = i + 1
            # print(
            #     f'CTCLabelConverter 들어옴 이닛함수 여기서 for i, char in enumerate(dict_character) char ### ??????? !!!!!!!!!!!!!!self.dict[char] : {self.dict[char]}')
            # if '✓' in char:
            #     # 'ๆ' 문자가 있을 경우
            #     print('=======================================')
            #     print("char 랑 같음char 랑 같음char 랑 같음char 랑 같음char 랑 같음char 랑 같음char 랑 같음")
            #     print('char 있음')
            # else:
            #     # 'ๆ' 문자가 없을 경우
            #     #print('char 없음')
            #     pass


        #print(f'CTCLabelConverter 이닛함수 !!!!!!!!!!!!!self.dict : {self.dict}')

        # if '✓' in self.dict:
        #     # 'ๆ' 문자가 있을 경우
        #     print('self.dict 있음')
        # else:
        #     # 'ๆ' 문자가 없을 경우
        #     print('self.dict 없음')

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)
        #print(f'CTCLabelConverter 이닛함수 !!!!!!!!!!!!!self.character : {self.character}')


    #def encode(self, text, batch_max_length=25):
    #def encode(self, text, batch_max_length=200):
    def encode(self, text, batch_max_length=300):
        # print(f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!text ( 라벨들 ) : {text}')
        # print(f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!batch_max_length ( 각 라벨의 최대 텍스트 길이 ( 300 으로 했음 ) ) : {batch_max_length}')
        # print(f'CTCLabelConverter encode 함수 들어옴 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!len(text)( 라벨들 총 개수 ) : {len(text)}')
        # if 'ๆ' in text:
        #     # 'ๆ' 문자가 있을 경우
        #     print('CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!text 있음')
        #     print('='*50)
        #     print('=있음 여기만 있음????왜??????' * 50)
        # else:
        #     # 'ๆ' 문자가 없을 경우
        #     print('CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!text 없음')


        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        #print(f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!length ( 각 라벨들의 텍스트 길이 배열 ) : {length}')

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        # print(
        #     f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!batch_text ( torch.LongTensor(len(text), batch_max_length).fill_(0) ) : {batch_text}')
        for i, t in enumerate(text):
            text = list(t)
            #print(f'CTCLabelConverter encode 함수 들어옴 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!공백 있음????text : {text}')
            # if '✓' in text:
            #     # 'ๆ' 문자가 있을 경우
            #     print('CTCLabelConverter encode 함수 들어옴 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!공백 있음????text 있음')
            #     print('=' * 200)
            #     print('=' * 200)
            #     print('=' * 200)
            #     print('있음 왜 여기만 있음??' * 10)
            #     print(f'CTCLabelConverter encode  함수 들어옴  for i, t in enumerate(text) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!text : {text}')
            #
            # else:
            #
            #
            #     print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ✓ 문자가 없을 경우 text : {text}')
            #     #'ๆ' 문자가 없을 경우
            #     print('CTCLabelConverter encode 함수 들어옴 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!공백 있음????text 없음')

            text = [self.dict[char] for char in text]
            # print(
            #     f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!text ( 라벨의 텍스트를 숫자로 변경 ) : {text }')
            # print(
            #     f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!torch.LongTensor(text) ( LongTensor로 변경 ) : {torch.LongTensor(text)}')

            batch_text[i][:len(text)] = torch.LongTensor(text)
            # print(
            #     f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!torch.LongTensor(text) ( LongTensor로 변경 ) : {torch.LongTensor(text)}')
            #
            # print(
            #     f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!batch_text.to(device) : {batch_text.to(device)}')
            # print(
            #     f'CTCLabelConverter encode 함수 들어옴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!torch.IntTensor(length).to(device) : {torch.IntTensor(length).to(device)}')

        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  preds_str = converter.decode(preds_index.data, preds_size.data) decode text_index : {text_index}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  preds_str = converter.decode(preds_index.data, preds_size.data) decode length : {length}')
        print(
            f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  preds_str = converter.decode(preds_index.data, preds_size.data) decode text_index.shape : {text_index.shape}')
        print(
            f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  preds_str = converter.decode(preds_index.data, preds_size.data) decode length.shape : {length.shape}')

        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):

            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode index : {index}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode l : {l}')


            t = text_index[index, :]
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode t : {t}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode text_index : {text_index}')
            # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode index : {index}')

            char_list = []
            for i in range(l):
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode i : {i}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode l : {l}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode t  : {t }')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode t[i] : {t[i]}')
                # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode t[i - 1] : {t[i - 1]}')

                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode len(char_list) : {len(char_list)}')
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode char_list : {char_list}')
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode i : {i}')
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode t : {t}')
                    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode t[i] : {t[i]}')
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode t[i - 1] : {t[i - 1]}')
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode self.character : {self.character}')
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode len(self.character) : {len(self.character)}')
                    # print(
                    #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode self.character[t[i]] 185 없어!!!!! : {self.character[t[i]]}')
                    char_list.append(self.character[t[i]])
                    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  decode self.character[t[i]] : {self.character[t[i]]}')

            text = ''.join(char_list)
            # print(
            #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode text : {text}')

            texts.append(text)
            # print(
            #     f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  char_list.append(self.character[t[i]]) 전 decode texts : {texts}')
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    #def encode(self, text, batch_max_length=25):
    #def encode(self, text, batch_max_length=200):
    def encode(self, text, batch_max_length=300):
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  encode text: {text}')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  encode batch_max_length: {batch_max_length}')
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
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  encode self.dict: {self.dict}')

        # 수정된 코드
        for char in text:
            try:
                char_value = self.dict[char]
                print(f"utils.py  encode self.dict[char] : {self.dict[char]}")
                # text = [self.dict[char] for char in text]
                print(f"utils.py  encode Character ( for char in text ): {char}, Value ( self.dict[char] ): {char_value}")
            except KeyError:
                print(f"utils.py  encode Character '{char}' not found in the dictionary.")



        text = [self.dict[char] for char in text]
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!utils.py  encode text: {text}')
        # 원래 코드
        # text = [self.dict[char] for char in text]




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

    #def encode(self, text, batch_max_length=25):
    def encode(self, text, batch_max_length=200):
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
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
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
