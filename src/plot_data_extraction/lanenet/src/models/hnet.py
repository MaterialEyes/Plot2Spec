import torch
import torch.nn as nn

from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class HNet(nn.Module):
    def __init__(self, BatchNorm=SynchronizedBatchNorm2d):
        """
        Args:
        """
        super(HNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = BatchNorm(16)

        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn2 = BatchNorm(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn3 = BatchNorm(32)

        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn4 = BatchNorm(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn5 = BatchNorm(64)

        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn6 = BatchNorm(64)

        self.ln1 = nn.Linear(8192, 1024)
        #self.conv7 = nn.Conv2d(64, 8, 1)
        #self.bn7 = BatchNorm(8)
        self.ln2 = nn.Linear(1024, 6)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        """
        Args: x input tensor
        Outputs:
            Coefficents of the transformation matrix (6 numbers)
        """

        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        x = self.conv6(x)
        x = self.relu(self.bn6(x))
        x = self.maxpool(x)

        #x = self.conv7(x)
        #x = self.relu(self.bn7(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.ln1(x))
        x = self.ln2(x)

        return x

def compute_hnet_loss(pts, coefs, order=3):
    """
    Compute hnet loss for set of line points

    Note:
        not finished yet.  Need to be careful checking how
        loss function is computed, esp. at padding lanes,
        and padding points

        Note:
            not finished yet.  Need to be careful checking how
            loss function is computed, esp. at padding lanes,
            and padding points
    """
    loss = 0
    for line_pts in pts:
        loss += __compute_hnet_loss(line_pts, coefs, order=order)
    return loss/len(pts)

def __compute_hnet_loss(pts, coefs, order=3):
    """
        pts (Nx2xM): set of points in the original plane
        coefs (Nx6): contains cooeficents of the H matrix

        order: polynomial order that used to fit the lane (2 or 3)
    """
    # adding 1to the 3rd dim of each point
    # import pdb; pdb.set_trace()
    pts = torch.cat((pts, torch.ones(pts.size(0), 1, pts.size(2)).type_as(coefs)), 1)

    # convert 6 cooeficents into H, which is a 3x3 matrix
    one = torch.ones(coefs.size(0), 1).type_as(coefs)
    x = torch.cat((coefs, one), 1)

    index_tensor = torch.tensor([0,1,2,4,5,7,8]).view(1, -1).repeat(coefs.size(0), 1).type_as(coefs).long()
    H = torch.zeros(coefs.size(0), 9).type_as(coefs).scatter_(1, index_tensor, x).reshape(-1, 3, 3)

    # Transform all points from orignal plan to the mapped plane using H
    pts_transformed = torch.matmul(H, pts)
    # Scale output so that the 3rd is 1
    pts_transformed = torch.div(pts_transformed, pts_transformed[:,2:3,:])

    # Fitting lines in the transformed space using the close-formed solution
    X_p = pts_transformed[:,0,:]
    Y_p = pts_transformed[:,1,:]

    #Y = torch.stack((Y_p*Y_p, Y_p, torch.ones(pts.size(1)))).t()
    if order == 2:
        Y = torch.stack((Y_p*Y_p, Y_p, torch.ones(pts.size(0), pts.size(2)).type_as(coefs)), 2) # Size: N x M x 3
    elif order == 3:
        Y = torch.stack((Y_p*Y_p*Y_p, Y_p*Y_p, Y_p, torch.ones(pts.size(0), pts.size(2)).type_as(coefs)), 2) # Size: N x M x 3
    else:
        raise ValueError('Unknown order', order)

    W = torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y.transpose(1,2), Y)), Y.transpose(1,2)), X_p.unsqueeze(2))

    # using the learned model to predict the x values in the projected plane
    X_pred = torch.matmul(Y, W) # Size: N x M x 1
    # concate with Y and ones vector to form a tensor size: N x M x 3
    pts_pred = torch.cat((X_pred, Y_p.unsqueeze(2), torch.ones(pts.size(0), pts.size(2), 1).type_as(coefs)), 2)
    # project it back to the original plane
    pts_back = torch.matmul(torch.inverse(H), pts_pred.transpose(1,2))
    # Scale output so that the 3rd is 1
    pts_back = torch.div(pts_back, pts_back[:,2:3,:])

    # import pdb; pdb.set_trace()
    # compute mse loss
    loss = torch.mean((pts[:,0,:] - pts_back[:,0,:])**2)
    return loss


if __name__ == "__main__":

    N = 10
    inputs = torch.rand(N, 3, 64, 128)  # N x C x H x W
    print("in.size():", inputs.size())
    model = HNet()
    coefs = model(inputs)
    pts = torch.rand(N, 2, 5)

    loss = hnet_loss(pts, coefs)
    print("out.size():", coefs.size())
    print("loss:", loss.item())
