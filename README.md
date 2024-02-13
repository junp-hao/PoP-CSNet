# Projected Landweber Optimization Based Deep Unfolding Network for Image Compressed Sensing (POP-CSNet) [PyTorch]

Abstract
--------

### Due to the powerful learning capability and fast processing speed of deep neural networks, a series of data-driven and deep unfolding networks for image reconstruction have emerged, achieving higher reconstruction quality. These reconstruction networks typically use convolutional neural network or residual neural network to extract high-dimensional features about the dominant structure component of the original image. However, the features from other dimensions about edge and texture details in the original image and losses generated at each iteration during the unfolding procedure are often neglected which would affect the quality of image reconstruction. In this paper, a Projected landweber Optimization based Progressive deep unfolding Compressive Sensing Network (PoP-CSNet) is proposed for image reconstruction. The PoP-CSNet designs a projected landweber block consisting of thresholding module (TSM) and progressive projecting module (PPM), along with a residual integration module (RIM). The PPM combines the approximate message passing algorithm with neural network to compute the projections of the optimal solution for original image, while the TSM effectively utilizes multi-dimensional image features through dense connection block, enhancing finer image details. Moreover, using RIM, image losses generated during optimization iterations are supplemented back into the reconstructed image. Finally, the effectiveness of PoP-CSNet is demonstrated on two standard benchmark datasets. Compared to classical compressive sensing image reconstruction networks, our network achieves higher reconstruction accuracy. Codes are available at **https://github.com/junp-hao/PoP-CSNet.**

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

![bird_004.png](./assets/bird_0.04.png)

![barbara_025.png](./assets/barbara_0.25.png)

![result](./images/test43_0.1.png)

![room_001.png](./assets/room_0.01.png)
