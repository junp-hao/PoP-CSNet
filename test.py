import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from utils.data_utils import numpy_to_img
from skimage.metrics import structural_similarity as ssim
from model.PoP_CSNet import PoP_CSNet


parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument("--cuda", default="True", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./results/CS_rate_0.25/net_best.pth", type=str,
                    help="model path")
parser.add_argument("--dataset", default="Test/Set11_mat", type=str, help="dataset name, Default: Set5")
parser.add_argument('--block_size', default=32, type=int, help='CS block size')
parser.add_argument('--sub_rate', default=0.25, type=float, help='sampling sub rate')


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = PoP_CSNet(opt.block_size, opt.sub_rate)


if opt.model != '':
    model.load_state_dict(torch.load(opt.model))

image_list = glob.glob(opt.dataset + "/*.*")

avg_psnr_predicted = 0.0
avg_ssim_predicted = 0.0
avg_elapsed_time = 0.0
img_id = 0

with torch.no_grad():
    for index, image_name in enumerate(image_list):
        print("Processing ", image_name)
        im_gt_y = sio.loadmat(image_name)['im_gt_y']  # 256*256
        img_id += 1

        im_gt_y = im_gt_y.astype(float)

        im_input = im_gt_y / 255.

        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

        model = model.cuda()
        im_input = im_input.cuda()

        start = time.time()

        res = model(im_input)
        end = time.time()
        elapsed_time = end - start
        avg_elapsed_time += elapsed_time

        if index == 0:
            avg_elapsed_time = 0

        res = res.cpu()

        im_res_y = res.data[0].numpy().astype(np.float32)

        im_res_y = numpy_to_img(im_res_y)

        psnr_predicted = PSNR(im_gt_y, im_res_y, shave_border=0)
        ssim_predicted = ssim(im_gt_y, im_res_y, data_range=255)
        print(psnr_predicted)

        avg_ssim_predicted += ssim_predicted
        avg_psnr_predicted += psnr_predicted

    print("It takes average {}s for processing".format(avg_elapsed_time / (len(image_list)-1)))

avg_psnr = avg_psnr_predicted / len(image_list)
avg_ssim = avg_ssim_predicted / len(image_list)
print("Dataset=", opt.dataset)
print("PSNR_predicted={},SSIM_predicted={}".format(avg_psnr,avg_ssim))

