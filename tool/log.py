import os
import logging


def get_log(name: str = "mot", out_dir: str = None) -> logging.Logger:
    logger = logging.getLogger(name)  # 子日志
    logger.setLevel(logging.INFO)  # INFO等级
    # 时间 - [等级] : 信息
    s_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s] : %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    # 控制台
    s_handler = logging.StreamHandler()
    s_handler.setFormatter(s_formatter)
    logger.addHandler(s_handler)

    # 输出到文件
    if out_dir is not None:
        assert os.path.isdir(out_dir), '输出文件路径错误'
        f_formatter = logging.Formatter(fmt='%(asctime)s : %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')
        f_handler = logging.FileHandler(out_dir, mode='w')
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)

    return logger


if __name__ == '__main__':
    fps = 30
    count = 1900
    logger = get_log('mot - 1')
    logger.info('read video complete.')
    logger.info('fps is {}, count is {}'.format(fps, count))
    logger.info('finish.')
