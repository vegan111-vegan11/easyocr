import torch.nn as nn
import os
import argparse
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
from utils import strLabelConverter
from torchvision import transforms
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./data/demo.png', help='the path to your image')
parser.add_argument('--model_path', type=str, default='./saved_models/english_g2.pth', help='path to pretrained model')
opt = parser.parse_args()

opt.cfg = Config.fromfile('./custom_example.yaml')
opt.alphabet = opt.cfg.data.character

converter = strLabelConverter(opt.alphabet)
transformer = dataset.resizeNormalize((100, 32))

model = crnn.CRNN(opt.cfg.model.num_fiducial, opt.cfg.model.input_channel, opt.cfg.model.hidden_size, opt.cfg.model.num_of_characters, opt.cfg.model.backbone, 1, opt.cfg.model.sensitive, opt.cfg.model.rgb)
if opt.model_path:
    print('loading pretrained model from %s' % opt.model_path)
    model.load_state_dict(torch.load(opt.model_path))

model.eval()

image = Image.open(opt.img_path).convert('L')
image = transformer(image)
if opt.cfg.data.sensitive:
    image = image.view(1, *image.size())
    image = Variable(image)
else:
    image = image.view(1, *image.size())
    image = Variable(image)
    image = image.to(device)
model = model.to(device)
model = model.eval()

with torch.no_grad():
    preds = model(image)
    _, preds_index = preds.max(2)
    preds_index = preds_index.view(-1)
    preds_str = converter.decode(preds_index.data, preds_size.data)

print('Predicted: %s' % preds_str)

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try: # multi gpu needs this
            self.rnn.flatten_parameters()
        except: # quantization doesn't work with this 
            pass
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
