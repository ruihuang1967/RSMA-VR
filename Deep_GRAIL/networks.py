from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from typing import NamedTuple

class CNN_Module(nn.Module):
    def __init__(self, n_hidden, p_dropout, out_channel, kernel_s, padding):
        super(CNN_Module, self).__init__()

        self.channel1 = out_channel
        self.channel2 = self.channel1
        self.channel3 = self.channel2 * 2
        self.RELU_slope = 0.3

        self.cnn1 = nn.Conv2d(in_channels=4, out_channels=self.channel1, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu1 = nn.LeakyReLU(negative_slope=self.RELU_slope)
        # self.do1 = nn.Dropout2d(p_dropout)

        self.cnn2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel2, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu2 = nn.LeakyReLU(negative_slope=self.RELU_slope)
        # self.do2 = nn.Dropout2d(p_dropout)

        self.cnn3 = nn.Conv2d(in_channels=self.channel2, out_channels=self.channel3, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu3 = nn.LeakyReLU(negative_slope=self.RELU_slope)
        # self.do3 = nn.Dropout2d(p_dropout)

        nn_in_fc = 480
        self.fc20 = nn.Linear(nn_in_fc, n_hidden)
        self.relu20 = nn.LeakyReLU(negative_slope=self.RELU_slope)
        # self.do20 = nn.Dropout(p_dropout)

        self.fc30 = nn.Linear(n_hidden, n_hidden)
        self.relu30 = nn.LeakyReLU(negative_slope=self.RELU_slope)
        # self.do30 = nn.Dropout(p_dropout)

    def forward(self, x): 

        out = self.cnn1(x)
        out = self.relu1(out)

        out = self.cnn2(out)
        out = self.relu2(out)

        out = self.cnn3(out)
        out = self.relu3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc20(out)
        out = self.relu20(out)

        out = self.fc30(out)
        out = self.relu30(out)

        return out

class CriticNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        action_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        Nt,
        N
    ):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_action = 1
        self.Nt = Nt
        self.N = N
        self.n_heads = 8
        self.decoder_layer = 1
        self.n_layers = n_layers
        self.RELU_slope = 0.001

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=self.RELU_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=self.RELU_slope),
            nn.Linear(hidden_dim, 1)
        )

        self.cnn_encoder = CNN_Module(hidden_dim, 0.1, 16, 2, 0)

    def forward(self, inputs):

        batch_size = inputs.size(0)
        node_embeddings = self.cnn_encoder(inputs)
        node_size = node_embeddings.size(1)
        a = self.value_head(node_embeddings)

        return a

class CriticNetwork_MLP(nn.Module): # Not used

    def __init__(
        self,
        input_dim,
        action_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        Nt,
        N
    ):
        super(CriticNetwork_MLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_action = 1
        self.Nt = Nt
        self.N = N
        step_context_dim = embedding_dim
        self.n_heads = 8
        self.decoder_layer = 1
        self.n_layers = n_layers
        self.RELU_slope = 0.001
    
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.MLP_module = nn.Sequential(*get_net(hidden_dim, self.n_layers))

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=self.RELU_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=self.RELU_slope),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # batch_size = state.size(0)                  # MLP Critic
        state = state.flatten(start_dim=1)
        action = action.flatten(start_dim=1)
        input_state = torch.cat([state, action], 1)

        a = F.relu(self.input_layer(input_state))
        a = self.MLP_module(a)
        a = self.value_head(a)

        return a


class ActorNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        action_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        Nt,
        N
    ):
        super(ActorNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_action = 1
        self.Nt = Nt
        self.N = N
        self.n_heads = 8
        self.decoder_layer = 1
        self.n_layers = n_layers

        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.cnn_encoder = CNN_Module(hidden_dim, 0.1, 16, 2, 0)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        node_embeddings = self.cnn_encoder(inputs)
        a = self.action_head(node_embeddings)
        a = self.max_action * a
        a = a.reshape(batch_size, self.Nt, self.N+1, 2)

        return a

    def noisy_forward(self, inputs):
        # Not implemented
        return None