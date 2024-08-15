# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Pedestrian detection system
File Name: window.py
Create Date: 2023/4/20
Description：图形化界面，可以检测摄像头、视频和图片文件
-------------------------------------------------
"""
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import threading
import cv2
import json
import sys
import os.path as osp
import os
import onnxruntime
import time


def load_single_img(img, reshape_size, mode='image', model=''):
    """
    Return：
        img0(H,W,C): 直接读取的图像矩阵
        img1(reshape_size,,C): 缩放过的图像矩阵
        img(1,C,reshepe_size): 经过处理，可直接输入给推理器
    """
    assert model == 'yolo' or model == 'fasterrcnn', 'Please use yolo or fasterrcnn'
    if model == 'yolo':
        mean = [0, 0, 0]
        std = [255.0, 255.0, 255.0]
    else:
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
    to_rgb = True
    if mode == 'video':
        img0 = img
    else:
        img0 = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
    img1 = cv2.resize(img0, reshape_size)  # (800,1333,3)
    img = img1.copy().astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    img = img.transpose(2, 0, 1)
    img = np.array(img)[np.newaxis, :, :, :]
    return img, img0, img1


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain_h = img1_shape[0] / img0_shape[0]
        gain_w = img1_shape[1] / img0_shape[1]  # gain  = old / new
        pad = [(img1_shape[1] - img0_shape[1] * gain_w) / 2,
               (img1_shape[0] - img0_shape[0] * gain_h) / 2]  # wh padding
    else:
        gain_h = ratio_pad[0][0]
        gain_w = ratio_pad[0][1]
        pad = ratio_pad[1]
    coords[[0, 2]] = (coords[[0, 2]] - pad[0]) / gain_w
    coords[[1, 3]] = (coords[[1, 3]] - pad[1]) / gain_h
    return coords



# 窗口主类
class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('行人检测系统')
        self.resize(800, 500)
        # 设置图标
        self.setWindowIcon(QIcon("images/UI/pedestrian.png"))
        # 图片读取进程
        self.output_size = 300
        self.img2predict = ""
        self.device = 'cuda'
        # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        ##
        self.init_setting()
        self.img_save_path = self.setting['img_save_path']
        self.vid_save_path = self.setting['vid_save_path']
        self.conf_thr = self.setting['conf_thr']
        self.person_name = self.setting['person_name']
        self.model = self.model_load(weights="models/yolo.onnx", device=self.device)  # todo 指明模型加载的位置的设备
        self.model_type = 'yolo'
        self.initUI()
        self.reset_vid()

    '''
    ***模型初始化***
    '''
    # @torch.no_grad()
    def model_load(self, weights, device='0'):
        if device == 'cpu':
            ort_session = onnxruntime.InferenceSession(weights, providers=['CPUExecutionProvider'])
            print('用了CPU')
        else:
            ort_session = onnxruntime.InferenceSession(weights, providers=['CUDAExecutionProvider'])   # 根据模型的扩展名如.onnx来构造推理器
            print('用了GPU')
        print("模型加载完成!")
        return ort_session

    def init_setting(self):
        """
        保存路径初始化
        """
        if os.path.exists('setting.json'):
            setting = open('setting.json', 'r', encoding ='utf-8')
            self.setting = json.load(setting)
        else:
            self.setting = {'img_save_path': "images/",
                            'vid_save_path': "images/",
                            'conf_thr': 0.45,
                            'person_name': "person"}
            with open('setting.json', 'w') as f:
                json.dump(self.setting, f)

    '''
    ***界面初始化***
    '''
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 14)
        font_main = QFont('楷体', 12)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片检测功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/waiting.png"))
        self.right_img.setPixmap(QPixmap("images/UI/waiting.png"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(100)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.open_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setIcon(QIcon("images/UI/up.png"))
        det_img_button.setIcon(QIcon("images/UI/start.png"))
        up_img_button.setStyleSheet("QPushButton{color:rgb(48,124,208)}"
                                    "QPushButton:hover{background-color: rgb(48,124,208);color: white}"
                                    "QPushButton{background-color:white}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:rgb(2,110,180)}"
                                     "QPushButton:hover{background-color:  rgb(48,124,208);color: white}"
                                     "QPushButton{background-color:white}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # 视频识别界面
        # 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("视频检测功能")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/waiting.png"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setIcon(QIcon("images/UI/capture.png"))
        self.mp4_detection_btn.setIcon(QIcon("images/UI/video.png"))
        self.vid_stop_btn.setIcon(QIcon("images/UI/stop.png"))
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:rgb(48,124,208)}"
                                                "QPushButton:hover{background-color: rgb(48,124,208);color: white}"
                                                "QPushButton{background-color:white}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:rgb(48,124,208)}"
                                             "QPushButton:hover{background-color: rgb(48,124,208);color: white}"
                                             "QPushButton{background-color:white}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:rgb(48,124,208)}"
                                        "QPushButton:hover{background-color: rgb(48,124,208);color: white}"
                                        "QPushButton{background-color:white}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_vid)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        # 添加组件到布局上
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        setting_widget = QWidget()
        setting_layout = QVBoxLayout()
        img_save_path_button = QPushButton("图片检测保存路径")
        img_save_path_button.setFont(font_main)
        img_save_path_button.clicked.connect(self.change_img_path)
        vid_save_path_button = QPushButton("视频检测保存路径")
        vid_save_path_button.setFont(font_main)
        vid_save_path_button.clicked.connect(self.change_vid_path)
        self.img_save_path_label = QLabel()
        self.img_save_path_label.setFont(font_main)
        self.img_save_path_label.setText(self.img_save_path)
        self.vid_save_path_label = QLabel()
        self.vid_save_path_label.setFont(font_main)
        self.vid_save_path_label.setText(self.vid_save_path)
        # 图片保存路径按钮和标签
        img_save_path_widget = QWidget()
        img_save_path_layout = QHBoxLayout()
        img_save_path_layout.addWidget(img_save_path_button, alignment=Qt.AlignLeft)
        img_save_path_layout.addStretch(2)
        img_save_path_layout.addWidget(self.img_save_path_label, alignment=Qt.AlignLeft)
        img_save_path_layout.addStretch(100)
        img_save_path_widget.setLayout(img_save_path_layout)
        img_save_path_button.setStyleSheet("QPushButton{color:rgb(48,124,208)}"
                                           "QPushButton:hover{background-color: rgb(48,124,208);color: white}"
                                           "QPushButton{background-color:white}"
                                           "QPushButton{border:2px}"
                                           "QPushButton{border-radius:5px}"
                                           "QPushButton{padding:5px 5px}"
                                           "QPushButton{margin:5px 5px}")
        self.img_save_path_label.setStyleSheet("QLabel{background-color:LightGrey}"
                                               "QLabel{border:2px}"
                                               "QLabel{border-radius:5px}"
                                               "QLabel{padding:5px 5px}"
                                               "QLabel{margin:5px 5px}")
        # 视频保存路径按钮和标签
        vid_save_path_widget = QWidget()
        vid_save_path_layout = QHBoxLayout()
        vid_save_path_layout.addWidget(vid_save_path_button, alignment=Qt.AlignLeft)
        vid_save_path_layout.addStretch(2)
        vid_save_path_layout.addWidget(self.vid_save_path_label, alignment=Qt.AlignLeft)
        vid_save_path_layout.addStretch(100)
        vid_save_path_widget.setLayout(vid_save_path_layout)
        vid_save_path_button.setStyleSheet("QPushButton{color:rgb(48,124,208)}"
                                           "QPushButton:hover{background-color: rgb(48,124,208);color: white}"
                                           "QPushButton{background-color:white}"
                                           "QPushButton{border:2px}"
                                           "QPushButton{border-radius:5px}"
                                           "QPushButton{padding:5px 5px}"
                                           "QPushButton{margin:5px 5px}")
        self.vid_save_path_label.setStyleSheet("QLabel{background-color:LightGrey}"
                                               "QLabel{border:2px}"
                                               "QLabel{border-radius:5px}"
                                               "QLabel{padding:5px 5px}"
                                               "QLabel{margin:5px 5px}")

        conf_thre_label = QLabel('检测框显示阈值')
        conf_thre_label.setFont(font_main)
        self.conf_thre_box = QDoubleSpinBox()
        self.conf_thre_box.setFont(font_main)
        self.conf_thre_box.setRange(0, 1)
        self.conf_thre_box.setMinimum(0.1)
        self.conf_thre_box.setMaximum(0.9)
        self.conf_thre_box.setSingleStep(0.1)
        self.conf_thre_box.setValue(self.conf_thr)
        conf_thre_label.setStyleSheet("QLabel{border:2px}"
                                      "QLabel{border-radius:5px}"
                                      "QLabel{padding:5px 5px}"
                                      "QLabel{margin:5px 5px}")
        self.conf_thre_box.setStyleSheet("QDoubleSpinBox{border:2px}"
                                         "QDoubleSpinBox{border-radius:5px}"
                                         "QDoubleSpinBox{padding:5px 5px}"
                                         "QDoubleSpinBox{margin:5px 5px}")
        conf_thre_widget = QWidget()
        conf_thre_layout = QHBoxLayout()
        conf_thre_layout.addWidget(conf_thre_label, alignment=Qt.AlignLeft)
        conf_thre_layout.addStretch(2)
        conf_thre_layout.addWidget(self.conf_thre_box, alignment=Qt.AlignLeft)
        conf_thre_layout.addStretch(100)
        conf_thre_widget.setLayout(conf_thre_layout)

        person_name_label = QLabel('检测框标签名称')
        person_name_label.setFont(font_main)
        self.person_name_edit = QLineEdit()
        self.person_name_edit.setPlaceholderText('请输入标签名称（仅支持英文）')
        self.person_name_edit.setFont(font_main)
        self.person_name_edit.setText(self.person_name)
        person_name_label.setStyleSheet("QLabel{border:2px}"
                                        "QLabel{border-radius:5px}"
                                        "QLabel{padding:5px 5px}"
                                        "QLabel{margin:5px 5px}")
        self.person_name_edit.setStyleSheet("QDoubleSpinBox{border:2px}"
                                            "QDoubleSpinBox{border-radius:5px}"
                                            "QDoubleSpinBox{padding:5px 5px}"
                                            "QDoubleSpinBox{margin:5px 5px}")
        person_name_widget = QWidget()
        person_name_layout = QHBoxLayout()
        person_name_layout.addWidget(person_name_label , alignment=Qt.AlignLeft)
        person_name_layout.addStretch(2)
        person_name_layout.addWidget(self.person_name_edit, alignment=Qt.AlignLeft)
        person_name_layout.addStretch(100)
        person_name_widget.setLayout(person_name_layout)
        # 设置确认按钮
        setting_confirm_button = QPushButton("保存设置")
        setting_confirm_button.setFont(font_main)
        setting_confirm_button.clicked.connect(self.save_setting)
        setting_confirm_button.setIcon(QIcon("images/UI/save.png"))
        setting_confirm_button.setStyleSheet("QPushButton{color:rgb(48,124,208)}"
                                             "QPushButton:hover{background-color: rgb(48,124,208);color: white}"
                                             "QPushButton{background-color:white}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")

        setting_layout.addWidget(img_save_path_widget)
        setting_layout.addStretch(2)
        setting_layout.addWidget(vid_save_path_widget)
        setting_layout.addStretch(2)
        setting_layout.addWidget(conf_thre_widget)
        setting_layout.addStretch(2)
        setting_layout.addWidget(person_name_widget)
        setting_layout.addStretch(100)
        setting_layout.addWidget(setting_confirm_button)
        setting_widget.setLayout(setting_layout)

        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')
        self.addTab(setting_widget, '设置')
        self.setTabIcon(0, QIcon('images/UI/picture.png'))
        self.setTabIcon(1, QIcon('images/UI/video.png'))
        self.setTabIcon(2, QIcon('images/UI/setting.png'))

    def change_img_path(self):
        """
        改变img保存路径
        """
        path = QFileDialog.getExistingDirectory(self, 'choose path', '')
        path = path + '/'
        self.img_save_path_label.setText(path)
        self.img_save_path = path

    def change_vid_path(self):
        """
        改变vid保存路径
        """
        path = QFileDialog.getExistingDirectory(self, 'choose path', '')
        path = path + '/'
        self.vid_save_path_label.setText(path)
        self.vid_save_path = path

    def save_setting(self):
        """
        确认设置改变
        """
        self.conf_thr = self.conf_thre_box.value()
        self.person_name = self.person_name_edit.text()
        self.setting['img_save_path'] = self.img_save_path
        self.setting['vid_save_path'] = self.vid_save_path
        self.setting['conf_thr'] = self.conf_thr
        self.setting['person_name'] = self.person_name

        with open('setting.json', 'w') as f:
            json.dump(self.setting, f)

    '''
    ***上传图片***
    '''
    def open_img(self):
        # 选择图像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)

            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]  # .shape[0]代表图片垂直尺寸
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)  # 这个（0，0）不懂！
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/waiting.png"))

    '''
    ***检测图片***
    '''
    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        conf_thres = self.conf_thr # confidence threshold 0.2 for yolo,0.5 for fasterrcnn
        names = [self.person_name, ]
        print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            # load image
            t1 = time.time()
            img, img0, img1 = load_single_img(source, reshape_size=(640, 480), model=self.model_type)
            print(img.shape)
            t2 = time.time()
            print('加载图片的时间:', t2-t1)
            # Inference
            dets, labels = model.run([model.get_outputs()[0].name, model.get_outputs()[1].name], {model.get_inputs()[0].name: img})
            t3 = time.time()
            print('推理单张图片的速度(FPS):', float(1/(t3-t2)))
            dets = np.squeeze(dets)
            labels = np.squeeze(labels)
            print(dets)
            # 选择大于阈值的框

            mask = dets[:, 4] >= conf_thres
            dets = dets[mask]
            labels = labels[mask]
            if len(dets) == 0:
                resize_scale = output_size / img0.shape[0]
                im0 = cv2.resize(img0, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result.jpg", im0)

                self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
                QMessageBox.warning(self, '提示', '未按测到行人')
            # Process predictions
            else:
                for det, label in zip(dets, labels):  # per image
                    # Rescale boxes from img_size to im0 size
                    det[:4] = scale_coords(img.shape[2:], det[:4], img0.shape).round()
                    [score, box] = [round(det[4], 2), det[:4]]
                    box = list(map(int, list(map(round, box))))
                    # print(box)
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], img0.shape[1])
                    ymax = min(box[3], img0.shape[0])  # notice
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = names[label] + ':' + str(score)
                    print(text)
                    cv2.rectangle(img0, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                    cv2.putText(img0, text, (xmin, int(ymin-5)), font, 0.5, (0, 128, 0), 1)
                if self.img_save_path:
                    cv2.imencode('.jpg', img0)[1].tofile(self.img_save_path + 'result_'+osp.basename(source))
                resize_scale = output_size / img0.shape[0]
                im0 = cv2.resize(img0, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result.jpg", im0)
                self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    '''
    ### 界面关闭事件 ### 
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示', "确认退出?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
            self.stopEvent.set()
        else:
            event.ignore()
            self.stopEvent.set()

    '''
    ### 视频关闭事件 ### 
    '''
    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''
    def open_vid(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    def load_single_video(self, source):
        cap = cv2.VideoCapture(source)  # 加载打开视频
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
        return cap, frames

    '''
    ### 视频开启事件 ### 
    '''
    # 视频和摄像头的主函数是一样的
    def detect_vid(self):
        model = self.model
        output_size = self.output_size
        conf_thres = self.conf_thr
        source = str(self.vid_source)
        webcam = self.webcam
        names = [self.person_name, ]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.vid_save_path+'output_vid.mp4', fourcc, 20.0, (1920, 1080))
        # Dataloader
        if webcam:
            cap, frames = self.load_single_video(0)
            print(frames)
            assert frames == -1, '读取错误,请检查摄像头'
        else:
            cap, frames = self.load_single_video(source)
        num_frame = 0
        # Run inference
        while True:
            ret, frame = cap.read()
            if webcam:
                frame = cv2.flip(frame, 1)
            print(frame.shape)
            t1 = time.time()  # 同步的时间t1
            img, img0, img1 = load_single_img(frame, mode='video', reshape_size=(640, 480), model=self.model_type)
            # cv2.imshow("image", frame)
            # Inference
            t2 = time.time()  # 同步的时间t2
            print('加载图片的时间:', t2-t1)
            dets, labels = model.run([model.get_outputs()[0].name, model.get_outputs()[1].name], {model.get_inputs()[0].name: img})
            t3 = time.time()  # 同步的时间t3
            print('推理单张图片的速度(FPS):', float(1/(t3-t2)))
            dets = np.squeeze(dets)
            labels = np.squeeze(labels)
            # 选择大于阈值的框
            mask = dets[:, 4] >= conf_thres
            dets = dets[mask]
            labels = labels[mask]
            if len(dets) == 0:
                frame = img0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
            else:
                for det, label in zip(dets, labels):  # per image
                    # Rescale boxes from img_size to im0 size
                    det[:4] = scale_coords(img.shape[2:], det[:4], img0.shape).round()

                    [score, box] = [round(det[4], 2), det[:4]]  # 置信度保留两位小数
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], img0.shape[1])
                    ymax = min(box[3], img0.shape[0])  # notice
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = names[label] + ':' + str(score)

                    cv2.rectangle(img0, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                    cv2.putText(img0, text, (xmin, int(ymin-5)), font, 1, (255, 255, 255), 1)
                    out.write(img0)
                    resize_scale = output_size / frame.shape[0]
                    frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                    self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
            if cv2.waitKey(25) & self.stopEvent.is_set()==True:
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                self.reset_vid()
                cap.release()
                break

    '''
    ### 界面重置事件 ### 
    '''
    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("images/UI/waiting.png"))
        self.vid_source = '0'
        self.webcam = True

    '''
    ### 视频重置事件 ### 
    '''
    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
