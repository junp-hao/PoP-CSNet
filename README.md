# A Projected Landweber Optimization Based Deep Unfolding Network for Image Compressed Sensing (POP-CSNet) [PyTorch]

Abstract
--------

### Due to the powerful learning capability and fastprocessing speed of deep neural networks, a series of data-driven and deep unfolding networks for image reconstructionhave emerged, achieving improved reconstruction quality. Thesereconstruction networks typically employ convolutional neuralnetworks or residual neural networks to extract high-dimensionalfeatures of the dominant structure component. However, the edgeand texture components in multi-dimensional features as well asthe image residuals generated at each iteration during the unfold-ing procedure are often neglected, which would affect the qualityof image reconstruction. In this paper, a Projected Landweber Optimization based Progressive deep unfolding Compressive Sensing Network (PoP-CSNet) is proposed for image compressivesensing reconstruction. A projected Landweber block (PL-Block)consisting of a thresholding module (TSM) and a progressiveprojecting module (PPM) is designed, along with a residualintegration module (RIM). The TSM utilizes the dense blockto fuse multi-dimensional image features, enhancing finer imagedetails effectively. The PPM combines the approximate messagepassing algorithm with deep neural networks to compute theprojections of the approximation solution for images. Moreover,using RIM, image residuals generated during each iterationstep are supplemented back into the reconstructed image. Theeffectiveness of PoP-CSNet is demonstrated on four standardbenchmark datasets, and comparisons with classical image com-pressive sensing reconstruction networks show that our networkcould achieve higher reconstruction accuracy. Codes are availableat: https://github.com/junp-hao/PoP-CSNet.

## Train

the training data file in `./datasets/train`, run:

```
python train.py
```

The model files will be in `./model`, respectively.

## Test

the model file `net_best.pth` in `./results`, run:

```shell
python test.py --testset_name=BSD68
```

The test sets are in `./test`.

## Results

![bird](./images/bird_0.04.png)

![barbara](./images/barbara_0.25.png)

![result](./images/test43_0.1.png)

![room](./images/room_0.01.png)
