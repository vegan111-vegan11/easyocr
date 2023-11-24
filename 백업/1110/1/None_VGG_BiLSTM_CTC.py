import torch
import torch.nn as nn

class VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG_FeatureExtractor, self).__init__()
        # Feature extraction layers (ConvNet)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size, hidden_size, num_classes):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x

class None_VGG_BiLSTM_CTC(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_classes):
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!None_VGG_BiLSTM_CTC 클래스 들어옴 input_channel : {input_channel}')


        super(None_VGG_BiLSTM_CTC, self).__init__()
        self.FeatureExtraction = VGG_FeatureExtractor()
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.Prediction = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.FeatureExtraction(x)

        # Sequence modeling
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.SequenceModeling(x)

        # Prediction
        x = self.Prediction(x)
        return x
