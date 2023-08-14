import hashlib
import os
import signal
import socket
import subprocess
import threading

import flask
from flask_socketio import SocketIO

import CIFAR10_utils
import CIFAR10_socket


class Display:

    def __init__(self):
        self.is_Board = False
        self.is_testing = False
        self.is_training = False
        self.processId = None
        self.app = flask.Flask(__name__, template_folder="templates", static_folder="static")
        self.socketio = SocketIO(self.app, async_mode='gevent')  # Initialize Flask-SocketIO

        self.app.add_url_rule("/", "/index/", methods=["GET", "POST"], view_func=self.index)
        self.app.add_url_rule("/identify", methods=["GET", "POST"], view_func=self.identify)
        self.app.add_url_rule("/upload", methods=["GET", "POST"], view_func=self.upload)
        self.app.add_url_rule("/train", methods=["GET", "POST"], view_func=self.train)
        self.app.add_url_rule("/trainModel", methods=["GET", "POST"], view_func=self.trainModel)
        self.app.add_url_rule("/closeTrainModel", methods=["GET", "POST"], view_func=self.closeTrainModel)
        self.app.add_url_rule("/testAcc", methods=["GET", "POST"], view_func=self.testAcc)
        self.app.add_url_rule("/startTestAcc", methods=["GET", "POST"], view_func=self.startTestAcc)
        self.app.add_url_rule("/closeTestAcc", methods=["GET", "POST"], view_func=self.closeTestAcc)
        self.app.add_url_rule("/tensorboard", methods=["GET", "POST"], view_func=self.tensorBoard)
        self.app.add_url_rule("/startBoard", methods=["GET", "POST"], view_func=self.startBoard)
        self.app.add_url_rule("/closeBoard", methods=["GET", "POST"], view_func=self.closeBoard)

    def index(self):
        return flask.render_template('index.html')

    def train(self):
        return flask.render_template('train.html')

    def testAcc(self):
        return flask.render_template('testAcc.html')

    def identify(self):
        return flask.render_template('identify.html')

    def tensorBoard(self):
        return flask.render_template('tensorboard.html')

    def trainModel(self):
        if flask.request.method == "POST":
            get_json = flask.request.json
            num_epoch = get_json["epochs"]
            lr_post = get_json["lr"]
            batch_post = get_json["batch"]
            if not self.is_training:
                # 创建进程并启动训练
                self.is_training = True
                self.socketio.emit("trainModel", "指令已发送，请耐心等待\r")
                training_process = threading.Thread(target=self.async_trainModel, args=(num_epoch,
                                                                                        lr_post,
                                                                                        batch_post))
                training_process.start()
            else:
                self.socketio.emit("trainModel", "模型正在训练，请勿重复运行\r")
        return 'Success'

    def async_trainModel(self, num_epoch, lr_post, batch_post):
        global roomid
        cifar10 = CIFAR10_socket.Cifar10()
        cifar10.trainModel(self.socketio, int(num_epoch), float(lr_post), int(batch_post))
        self.is_training = False

    def closeTrainModel(self):
        if flask.request.method == "POST":
            CIFAR10_socket.lock = 1
            self.is_training = False
        return 'Success'

    def startTestAcc(self):
        if flask.request.method == "POST":
            if not self.is_testing:
                # 创建进程并启动训练
                self.is_testing = True
                self.socketio.emit("testAcc", "指令已发送，请耐心等待\r")
                training_process = threading.Thread(target=self.async_testAcc, args=())
                training_process.start()
            else:
                self.socketio.emit("testAcc", "模型正在评估，请勿重复运行\r")
        return 'Success'

    def async_testAcc(self):
        cifar10 = CIFAR10_socket.Cifar10()
        cifar10.testModel(self.socketio, "testAcc")
        self.is_testing = False

    def closeTestAcc(self):
        if flask.request.method == "POST":
            CIFAR10_socket.lockAcc = 1
            self.is_training = False
        return 'Success'

    def startBoard(self):
        if flask.request.method == "POST":
            if not self.is_Board:
                # 创建进程并启动训练
                self.is_Board = True
                board_process = threading.Thread(target=self.async_Board(), args=())
                board_process.start()
                self.socketio.emit("Board",
                                   "Serving TensorBoard on localhost; to expose to the network, use a proxy or pass "
                                   "--bind_all\rTensorBoard 2.12.3 at http://localhost:6006/")
            else:
                self.socketio.emit("Board", "面板正在运行，请勿重复运行\r")
        return 'Success'

    def async_Board(self):
        process = subprocess.Popen(["tensorboard", "--logdir=runs"])
        self.processId = process.pid

    def closeBoard(self):
        if flask.request.method == "POST":
            if self.is_Board:
                os.kill(self.processId, signal.SIGTERM)
                self.is_Board = False
                self.socketio.emit("Board", "\r可视化面板已关闭\r")
        return 'Success'

    def upload(self):
        if flask.request.method == "POST":
            file = flask.request.files['file']
            # 处理上传的文件，例如保存到磁盘或进行其他操作
            # 计算文件的 MD5 值
            md5_hash = hashlib.md5()
            while True:
                chunk = file.read(4096)
                if not chunk:
                    break
                md5_hash.update(chunk)
            file.seek(0)
            md5 = md5_hash.hexdigest()

            # 获取文件扩展名
            _, ext = os.path.splitext(file.filename)

            # 生成新的文件名
            new_filename = md5 + ext

            # 保存文件
            savePath = os.path.join('static/upload', new_filename)
            file.save(savePath)
            d = CIFAR10_utils.useModel(image_path=savePath)
            return d
        return "403"

    def run(self):
        # 获取本机 IP 地址
        ip = socket.gethostbyname(socket.gethostname())
        print(f"Server running at: http://{ip}:5002/")
        print(f"                   http://127.0.0.1:5002/")

        self.socketio.run(self.app, host='0.0.0.0', port=5002, debug=False)


if __name__ == '__main__':
    app = Display()
    app.run()
