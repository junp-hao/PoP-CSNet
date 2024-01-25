from torch import nn
from torchsummary import summary
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


# Reshape layer
class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0

    def __init__(self, block_size):
        Reshape_Concat_Adap.blocksize = block_size

    @staticmethod
    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)

        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize), int(h_ * Reshape_Concat_Adap.blocksize))).cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))
                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]
                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)


def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)


# DenseLayer
class DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """

    def __init__(self, num_input_features, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.PReLU())
        self.add_module("conv1", nn.Conv2d(num_input_features, 4 * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(4 * growth_rate))
        self.add_module("relu2", nn.PReLU())
        self.add_module("conv2", nn.Conv2d(4 * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


# DenseBlock
class DenseBlock(nn.Sequential):
    def __init__(self, growth_rate):
        super(DenseBlock, self).__init__()
        self.layer1 = DenseLayer(64, growth_rate)
        self.layer2 = DenseLayer(64 + 16, growth_rate)
        self.layer3 = DenseLayer(64 + 16 + 16, growth_rate)
        self.layer4 = DenseLayer(64 + 16 + 16 + 16, growth_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Transition
class Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""

    def __init__(self):
        super(Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(128))
        self.add_module("relu", nn.PReLU())
        self.add_module("conv", nn.Conv2d(128, 64,
                                          kernel_size=1, stride=1))

    def forward(self, x):
        x = super(Transition, self).forward(x)
        return x


# Deblock
class Deblocker(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),

                               nn.PReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.PReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.PReLU(),
                               nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, inputs):
        output = self.D(inputs)
        return output


#  code of PoP-CSNet_Enhanced (Enhanced version of CSNet)
class PoP_CSNet(nn.Module):
    def __init__(self, blocksize=32, subrate=0.1):
        super(PoP_CSNet, self).__init__()
        self.blocksize = blocksize
        self.lambda_step1 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step2 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step3 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step4 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step5 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step6 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step7 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step8 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step9 = nn.Parameter(torch.Tensor([0.5]))

        self.lambda_res_step1 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_res_step2 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_res_step3 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_res_step4 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_res_step5 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_res_step6 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_res_step7 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_res_step8 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_res_step9 = nn.Parameter(torch.Tensor([0.5]))

        self.soft_thr1 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr2 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr3 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr4 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr5 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr6 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr7 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr8 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr9 = nn.Parameter(torch.Tensor([0.01]))

        self.lambda_step_z1 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step_z2 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step_z3 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step_z4 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step_z5 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step_z6 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step_z7 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step_z8 = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step_z9 = nn.Parameter(torch.Tensor([0.5]))

        # for sampling
        self.sampling = nn.Conv2d(1, int(np.round(blocksize * blocksize * subrate)), blocksize, stride=blocksize,
                                  padding=0, bias=False)
        # upsampling
        self.upsampling = nn.Conv2d(int(np.round(blocksize * blocksize * subrate)), blocksize * blocksize, 1, stride=1,
                                    padding=0, bias=False)

        # reconstruction network
        self.up_chnnel = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )

        self.down_chnnel = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.same_chnnel = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.transition = nn.Sequential(
            DenseBlock(16),
            Transition(),
        )

        self.deblock = Deblocker()

    def forward(self, x):
        y = self.sampling(x)
        x = self.upsampling(y)
        init_x = My_Reshape_Adap(x, self.blocksize)  # Reshape + Concat

        z0 = y - self.sampling(init_x)

        up_x1 = self.up_chnnel(init_x)
        dense_x1 = self.transition(up_x1) + up_x1
        update_x1, res_x1, z1 = self.update_x(dense_x1, y, self.lambda_step1, self.lambda_step_z1, self.soft_thr1, z0)

        up_x2 = self.up_chnnel(update_x1)
        dense_x2 = self.transition(up_x2) + up_x2
        update_x2, res_x2, z2 = self.update_x(dense_x2, y, self.lambda_step2, self.lambda_step_z2, self.soft_thr2, z1)

        up_x3 = self.up_chnnel(update_x2)
        dense_x3 = self.transition(up_x3) + up_x3
        update_x3, res_x3, z3 = self.update_x(dense_x3, y, self.lambda_step3, self.lambda_step_z3, self.soft_thr3, z2)

        up_x4 = self.up_chnnel(update_x3)
        dense_x4 = self.transition(up_x4) + up_x4
        update_x4, res_x4, z4 = self.update_x(dense_x4, y, self.lambda_step4, self.lambda_step_z4, self.soft_thr4, z3)

        up_x5 = self.up_chnnel(update_x4)
        dense_x5 = self.transition(up_x5) + up_x5
        update_x5, res_x5, z5 = self.update_x(dense_x5, y, self.lambda_step5, self.lambda_step_z5, self.soft_thr5, z4)

        up_x6 = self.up_chnnel(update_x5)
        dense_x6 = self.transition(up_x6) + up_x6
        update_x6, res_x6, z6 = self.update_x(dense_x6, y, self.lambda_step6, self.lambda_step_z6, self.soft_thr6, z5)

        up_x7 = self.up_chnnel(update_x6)
        dense_x7 = self.transition(up_x7) + up_x7
        update_x7, res_x7, z7 = self.update_x(dense_x7, y, self.lambda_step7, self.lambda_step_z7, self.soft_thr7, z6)

        up_x8 = self.up_chnnel(update_x7)
        dense_x8 = self.transition(up_x8) + up_x8
        update_x8, res_x8, z8 = self.update_x(dense_x8, y, self.lambda_step8, self.lambda_step_z8, self.soft_thr8, z7)

        up_x9 = self.up_chnnel(update_x8)
        dense_x9 = self.transition(up_x9) + up_x9
        update_x9, res_x9, z9 = self.update_x(dense_x9, y, self.lambda_step9, self.lambda_step_z9, self.soft_thr9, z8)

        final_x = self.up_chnnel(update_x9)
        final_x = self.same_chnnel(final_x)
        final_x = self.down_chnnel(final_x)

        final_x = final_x + self.lambda_res_step1 * res_x1 + self.lambda_res_step2 * res_x2 + self.lambda_res_step3 * res_x3 + \
                  self.lambda_res_step4 * res_x4 + self.lambda_res_step5 * res_x5 + self.lambda_res_step6 * res_x6 + \
                  self.lambda_res_step7 * res_x7 + self.lambda_res_step8 * res_x8 + self.lambda_res_step9 * res_x9
        x_final = final_x - self.deblock(final_x)

        return x_final

    def update_x(self, x, y, lambda_step_x, lambda_step_z, soft_thr, z):

        x = self.same_chnnel(x)
        x = torch.mul(torch.sign(x), F.relu(torch.abs(x) - soft_thr))
        x = self.down_chnnel(x)

        res_y = y - self.sampling(x)
        z_new = res_y + lambda_step_z * z
        res_x = My_Reshape_Adap(self.upsampling(res_y), self.blocksize)
        res_z = My_Reshape_Adap(self.upsampling(z_new), self.blocksize)
        x = x + lambda_step_x * res_z

        return x, res_x, z_new
