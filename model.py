import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden2word = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # ignore last output, since last input character is <end>
        # so, we should get <end>, <end> at the end of output
#        hidden = self.init_hidden(self.num_layers, len(captions), self.hidden_size)

        embed = self.embed(captions[:,:-1])

        embed = torch.cat((features.unsqueeze(1), embed), 1)

        lstm_out, _ = self.lstm(embed)

        # get the scores for the most likely tag for a word
        tag_outputs = self.hidden2word(lstm_out)
        return tag_outputs
#        print(tag_outputs.shape)
#        tag_scores = F.log_softmax(tag_outputs, dim=2)

#        return tag_scores

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

#        hidden = self.init_hidden(self.num_layers, 1, self.hidden_size)

        result = []

#        print('sample is called')
#        print(inputs.shape)
        for i in range(max_len):
#            print(lstm_out)
            lstm_out, states = self.lstm(inputs, states)
            tag_output = self.hidden2word(lstm_out)
#            print(tag_output)
#            print(tag_output.shape)

#            tag_score = F.log_softmax(tag_output, dim=2)
#            print(tag_score)
#            print(tag_score.shape)

            predicted = torch.argmax(tag_output, dim=-1)
#            print(predicted)
#            print(predicted.shape)
            result.append(predicted[0,0])
            inputs = self.embed(predicted)
        return result

    def init_hidden(self, num_layers, batch_size, hidden_size):
        ''' At the start of training, we need to initialize a hidden state;
            there will be none because the hidden state is formed based on perviously seen data.
            So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))
        return  hidden
