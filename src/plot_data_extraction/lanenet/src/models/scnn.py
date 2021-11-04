import torch
import torch.nn as nn


class SCNN(nn.Module):
    def __init__(self, kernel_width=9,
                 directions=['D', 'U', 'L', 'R']):
        """
        Args:
            kernel_width: size of the kernel width in the paper
        """
        super(SCNN, self).__init__()
        pad_size = int(kernel_width/2)

        self.directions = directions
        if 'D' in self.directions:
            self.conv_down = nn.Conv2d(
                1, 1, kernel_size=(
                    1, kernel_width), padding=(
                    0, pad_size))
        if 'U' in self.directions:
            self.conv_up = nn.Conv2d(
                1, 1, kernel_size=(
                    1, kernel_width), padding=(
                    0, pad_size))
        if 'L' in self.directions:
            self.conv_left = nn.Conv2d(
                1, 1, kernel_size=(
                    1, kernel_width), padding=(
                    0, pad_size))
        if 'R' in self.directions:
            self.conv_right = nn.Conv2d(
                1, 1, kernel_size=(
                    1, kernel_width), padding=(
                    0, pad_size))

        self.relu = nn.ReLU(inplace=True)

    def __scnn(self, x, conv):
        """
        SCNN on a single direction, assuming the tensor is already in the right
        shape, e.g. NxHxWxC when applying SCNN in the 'D' direction

        Inputs:
            x: input tensor
            conv: the conv layer that will be used

        Returns:
            output

        """
        out = x.clone()
        H = x.size()[1]
        for i in range(1, H):
            out_i = self.relu(conv(out[:, i-1, :, :].unsqueeze(1)))
            out_i = torch.add(out_i.squeeze(1), x[:, i, :, :])
            out[:, i, :, :] = out_i

        return out

    def forward(self, x):
        # top to down
        # N x C x H x W --> N x H x W x C
        out = x.permute(0, 2, 3, 1)
        out = self.relu(self.__scnn(out, self.conv_down))

        # down to top
        # N x H x W x C --> N x H x W x C
        out = torch.flip(out, [1])
        out = self.relu(self.__scnn(out, self.conv_up))

        # left to right
        # N x H x W x C --> N x W x H x C
        out = out.permute(0, 2, 1, 3)
        out = self.relu(self.__scnn(out, self.conv_left))

        # right to left
        out = torch.flip(out, [1])
        out = self.relu(self.__scnn(out, self.conv_right))

        # N x W x H x C --> N x C x H x W
        out = out.permute(0, 3, 2, 1)
        return out


if __name__ == "__main__":

    inputs = torch.rand(8, 128, 100, 112)  # N x C x H x W
    print("in.size():", inputs.size())
    model = SCNN()
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
    outputs = model(inputs)
    print("out.size():", outputs.size())
