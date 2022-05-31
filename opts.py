import os
import yaml
import argparse
from easydict import EasyDict as edict


class YamlParser(edict):
    """
    通过EasyDict解析默认配置文件
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


class opts:
    def __init__(self):

        self.parser = argparse.ArgumentParser(description='Strong Sort')

        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )

        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )

        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )

        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching'
        )

        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )

        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )

        self.parser.add_argument(
            '--config_path',
            default='./configs/config.yaml',
            help='default config file path'
        )

    def parse(self, args=''):
        if args == '':
            opti = self.parser.parse_args()  # 默认deep sort
        else:
            opti = self.parser.parse_args(args)
        opti.cfg = YamlParser(config_file=opti.config_path)  # 加载配置文件
        opti.min_confidence = opti.cfg.DET.CONF_THRES
        opti.nms_max_overlap = 1.0
        opti.min_detection_height = opti.cfg.DET.MIN_HEIGHT
        opti.max_cosine_distance = opti.cfg.MOT.MAX_DIST

        if opti.MC:
            opti.max_cosine_distance += 0.05

        if opti.EMA:
            opti.nn_budget = 1
        else:
            opti.nn_budget = opti.cfg.MOT.BUDGET

        return opti


opt = opts().parse()
print(opt)
