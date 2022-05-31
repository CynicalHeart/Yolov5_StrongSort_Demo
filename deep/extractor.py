import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from deep.baseline import Baseline


class Extractor(object):
    def __init__(self, cfg, use_cuda=True):
        self.net = Baseline(cfg.REID.NUM_CLASS, cfg.REID.LAST_STRIDE, None,
                            cfg.REID.NECK, cfg.REID.NECK_FEAT, cfg.REID.NAME, cfg.REID.PRETRAIN)
        self.net.load_param(cfg.REID.PRETRAIN_PATH)
        self.device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)
        self.size = (128, 256)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        预处理
        TODO:
            1. to float with scale from 0 to 1 归一化
            2. resize to (128, 256) as Market1501 dataset did  转 128,256
            3. concatenate to a numpy array => numpy
            3. to torch Tensor  => Tensor
            4. normalize  标准化
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)  # -> numpy(float) -> 0~1 -> 128,256

        # im_crops: List[cv.Mat], 多出一个维度B, 再cat组成批次 [B,C,H,W] : torch.Tensor-float
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)  # 输入送入GPU
            features = self.net(im_batch)  # 获取特征集合
        return features.cpu().numpy()  # 从GPU->CPU->numpy数组


if __name__ == '__main__':
    from opts import opt
    config = opt.cfg

    img_p1 = 'E:/MySpace/resource/dataset/market1501/bounding_box_train/0002_c1s1_000776_01.jpg'
    img_p2 = 'E:/MySpace/resource/dataset/market1501/bounding_box_train/0002_c1s1_000801_01.jpg'
    img_1 = cv2.imread(img_p1)[:, :, (2, 1, 0)]  # H W C 通道BGR->RGB
    img_2 = cv2.imread(img_p2)[:, :, (2, 1, 0)]
    extr = Extractor(config)
    f1 = extr([img_1])
    f2 = extr([img_2])
    f1 = f1 / np.linalg.norm(f1, axis=1, keepdims=True)
    f2 = f2 / np.linalg.norm(f2, axis=1, keepdims=True)

    print(1. - np.dot(f1, f2.T))
