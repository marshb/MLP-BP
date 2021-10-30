import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, num_patch,
                 depth, token_dim, channel_dim, dropout=0.2):
        super().__init__()

        self.num_patch = num_patch
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels, dim, kernel_size=1, stride=1),
            Rearrange('b c t -> b t c'),
        )

        # todo: LSTM emb
        self.lstm_patch_emb = nn.Sequential(
            Rearrange('b c t -> b t c'),
            nn.LSTM(input_size=in_channels, hidden_size=int(0.5*dim), num_layers=1,
                                      bidirectional=True, batch_first=True),
        )

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.conv1d_decode = nn.Sequential(
            nn.Conv1d(num_patch, 2*num_patch, kernel_size=6, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv1d(2*num_patch, 4*num_patch, kernel_size=6, stride=4, padding=1),
            nn.ReLU(),
            # nn.Conv1d(512, 512, kernel_size=6, stride=4, padding=1)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(4*num_patch, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x = self.to_patch_embedding(x)
        x, (hn, cn) = self.lstm_patch_emb(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.conv1d_decode(x)
        x = x.mean(dim=2)
        return self.mlp_head(x)


if __name__ == "__main__":
    img = torch.ones([4, 12, 256])
    dropout = 0.2

    model = MLPMixer(in_channels=12, num_patch=256, num_classes=2,
                     dim=128, depth=6, token_dim=512, channel_dim=512, dropout=dropout)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)   # [B, in_channels, image_size, image_size]

