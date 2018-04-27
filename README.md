# tvNetPytorch
Pytorch implementation of [tvNet](https://github.com/LijieFan/tvnet) by [LijieFan](https://github.com/LijieFan)

The original implementation is by tensorflow and most of the code this Pytorch implementation is adapted from that one

# Sample
Note that the code of optic flow visualization is from [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) by [ClementPinard](https://github.com/ClementPinard), the visualization scheme seems different from the one provided in the original implementation.

| frame1 | frame2 | prediction |
|--------|--------|------------|
| <img src='img1.png' width=256> | <img src='img2.png' width=256> | <img src='flow.png' width=256> |

# Reference
    @inproceedings{fan2018end,
    title={End-to-End Learning of Motion Representation for Video Understanding},
    author={Fan, Lijie and Huang, Wenbing and Gan, Chuang and Ermon, Stefano and Gong, Boqing and Huang, Junzhou},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={},
    year={2018}
	}

