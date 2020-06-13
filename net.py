import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter





class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self._name = 'Flatten'

    def forward(self, inputs):
        return inputs.reshape(inputs.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self._name = 'Reshape'
        self.out_shape = out_shape

    def forward(self, inputs):
        return inputs.reshape((inputs.size(0),) + self.out_shape)



# https://github.com/neale/Adversarial-Autoencoder/blob/master/generators.py
class Decoder(nn.Module):
    # can't turn dropout off completely because otherwise the loss -> NaN....
    # batchnorm does not seem to help things...
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self._name = 'decoder'
        self.hidden_dim = hidden_dim
        self.dropoutrate = 0.05

        self.input_layers = {}

        self.time_values = ['wday', 'month', 'event_name_1_num', 'event_type_1_num', 'event_name_2_num', 'event_type_2_num', 'wk', 'sell_price']


        self.rnn = nn.GRU(len(self.time_values), self.hidden_dim, num_layers=1, batch_first=True,
                   bidirectional=False)  # nn.Dropout(p=self.dropout)
        self.linear = nn.Linear(self.hidden_dim, 1)



    def forward(self, inputs, hidden_input):
        input_ts = torch.cat([inputs[inp].float().unsqueeze(-1) for inp in self.time_values], dim=-1)

        mb_size = input_ts.size(0)
        n_steps = input_ts.size(1)
        #print(hidden_input.size())
        #print(input_ts.size())

        output, hidden_output = self.rnn(input_ts, hidden_input.unsqueeze(0))
        output = self.linear(output.reshape(mb_size*n_steps, output.size(2)))
        output = F.selu(output)
        output = output.reshape(mb_size, n_steps)



        #print(output.size())
        #print(hidden_output.size())


        return output




# https://github.com/neale/Adversarial-Autoencoder/blob/master/generators.py
class Encoder(nn.Module):
    # can't turn dropout off completely because otherwise the loss -> NaN....
    # batchnorm does not seem to help things...
    def __init__(self, hidden_dim, cat2val):
        super(Encoder, self).__init__()
        self._name = 'encoder'
        self.hidden_dim = hidden_dim
        self.dropoutrate = 0.05

        self.input_layers = {}

        self.cat2val = cat2val
        self.meta_values = self.cat2val.keys() #inputs for the hidden parameter
        self.time_values = ['wday', 'month', 'event_name_1_num', 'event_type_1_num', 'event_name_2_num', 'event_type_2_num', 'wk', 'sell_price', 'sale_counts']


        for inp in self.meta_values:
            self.input_layers[inp] = nn.Embedding(len(cat2val[inp]), self.hidden_dim)
            #self.input_layers[inp].apply(weights_init_linear)
            self.add_module(inp + '_input', self.input_layers[inp])


        self.rnn = nn.GRU(len(self.time_values), self.hidden_dim, num_layers=1, batch_first=True,
                   bidirectional=False)  # nn.Dropout(p=self.dropout)





    def forward(self, inputs):
        meta_inputs = []

        for inp in self.meta_values:
            #x = torch.tensor([inputs[i][inp] for i in range(len(inputs))])
            x = self.input_layers[inp](inputs[inp])
            meta_inputs.append(x)


        #not clear whether summing or concatenating the meta inputs is better
        hidden_input = torch.sum(torch.stack(meta_inputs, dim=0), dim=0).unsqueeze(0)

        input_ts = torch.cat([inputs[inp].float().unsqueeze(-1) for inp in self.time_values], dim=-1)

        #print(hidden_input.size())
        #print(input_ts.size())

        output, hidden_output = self.rnn(input_ts, hidden_input)

        #print(output.size())
        #print(hidden_output.size())


        return hidden_output.squeeze(0)




