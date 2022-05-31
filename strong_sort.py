import os
import time
import cv2
from typing import List

from opts import opt
from sort.tracker import Tracker
from deep.detection import Detection
from deep.deep_handler import Handler
from tool.draw import draw_dets, draw_tracks
from sort.nn_matching import NearestNeighborDistanceMetric

# ==== CPU配置 ====
cpu_count: int = os.cpu_count()  # cpu数量
os.environ["OMP_NUM_THREADS"] = str(cpu_count)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
os.environ["MKL_NUM_THREADS"] = str(cpu_count)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)

# ==== 全局变量 ====
handler = Handler()  # 检测、特征处理器
metrics = NearestNeighborDistanceMetric('cosine', opt.cfg.MOT.MAX_DIST)  # 度量
vid_format = ['avi', 'mp4', 'm4v', 'wmv', 'mkv', 'mpeg', 'mov']  # 常见视频格式

# ==== RES ====
output_path = opt.cfg.RES.OUTPUT  # 输出路径
vid_writer = None  # 视频写入
txt_path = None  # 跟踪文件存储

if __name__ == '__main__':
    # 1. 获取视频列表或者直接读取视频
    source: str = opt.cfg.DET.SOURCE
    input_list = []  # 路径列表
    if os.path.isdir(source):
        # 获取全部目录下文件 && 筛选正确尾缀
        file_names = os.listdir(source)
        for f_name in file_names:
            ext = f_name.split('.')[-1]  # 视频后缀
            p = os.path.join(source, f_name)  # 视频全路径
            if os.path.isfile(p) and ext in vid_format:
                input_list.append(p)
    elif os.path.isfile(source):
        # 文件 && 判断ext
        ext = source.split('.')[-1]  # 视频后缀
        if ext in vid_format:
            input_list.append(source)
        else:
            raise FileNotFoundError('文件格式不合规')
    else:
        raise FileNotFoundError('输入为文件或是目录')

    # 2. 遍历集合, 进行跟踪
    for vid, vp in enumerate(input_list, start=1):
        # 3. 获取基础信息
        file_name = os.path.basename(vp).split('.')[0]
        window_name = "Demo-{0} {1}".format(vid, file_name)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # 窗口
        cap = cv2.VideoCapture(vp)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # 平均帧率

        print(
            'video {}, width: {}, height: {}, fps is {:.1f}, total frames: {}'.format(file_name, frame_width,
                                                                                      frame_height,
                                                                                      fps, frame_count))

        if opt.cfg.DET.SAVE_VID:
            video_path = os.path.join(output_path, '%s.mp4' % window_name)
            vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (frame_width, frame_height))
        if opt.cfg.DET.SAVE_TEXT:
            txt_path = os.path.join(output_path, '%s.txt' % window_name)

        tracker = Tracker(metrics)  # 初始化轨迹
        start = time.time()
        # 4. 读取视频, 进行跟踪
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            # frame = cv2.resize(frame, (frame_width, frame_height))
            img = frame[:, :, ::-1]  # bgr -> rgb 存入img
            # 获取List[Detection]
            dets: List[Detection] = handler(img)  # 输入处理后的图像

            tracker.predict()  # 预测
            output = tracker.update(dets)  # 更新

            # 5. 可视化检测和轨迹
            frame = draw_dets(frame, dets)
            frame = draw_tracks(frame, output)

            #  6. 保存轨迹至文件
            if opt.cfg.DET.SAVE_TEXT and len(output) != 0:
                for label, info in output.items():
                    x1, y1, x2, y2 = [int(e) for e in info[0:4]]  # 坐标信息
                    conf = info[4]
                    with open(txt_path, 'a') as f:
                        f.write('{}, {}, {}, {}, {}, {}, {:.6f}, -1, -1, -1'
                                .format(frame_idx, label, x1, y1, x2, y2, conf) + '\n')

            #  7. 展示视频
            if opt.cfg.DET.SHOW_VID:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            #  8. 保存视频结果
            if opt.cfg.DET.SAVE_VID:
                vid_writer.write(frame)  # 存储

        end = time.time()
        print('it takes {:.1f} seconds, sort\'s fps is: {:.2f}.'.format(end - start,
                                                                        (frame_count / (end - start))))
        print('The last identity is %d' % (tracker._next_id - 1) + '\n')

        # 9. 释放资源
        cap.release()
        if opt.cfg.DET.SAVE_VID:
            vid_writer.release()
        metrics.samples = {}  # 清空samples存储的轨迹

        # 取消窗口
        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()
