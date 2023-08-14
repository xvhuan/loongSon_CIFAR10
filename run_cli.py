import os

import CIFAR10_utils


def main_menu():
    os.system('cls')
    print("===主界面===")
    print("1.模型训练")
    print("2.准度测试")
    print("3.模型推理")
    print("4.训练数据可视化")
    print("5.退出")


def train_model():
    os.system('cls')
    epoch = int(input("请输入迭代次数"))
    lr = float(input("请输入初始学习率"))
    batch_size = int(input("请输入批量大小"))
    CIFAR10_utils.Cifar10().trainModel(num_epochs=epoch, lr=lr, batch_size=batch_size)
    input("按任意键返回主菜单...")

def accuracy_test():
    os.system('cls')
    CIFAR10_utils.Cifar10().testModel()
    input("按任意键返回主菜单...")


def visualize_training_data():
    os.system('cls')
    print("正在启动可视化面板...")
    os.system("tensorboard --logdir=runs")
    input("按任意键返回主菜单...")


def model_use():
    os.system("cls")
    imagePath = input("请输入图片路径：")
    modelPath = input("请输入模型路径，空则默认：")
    print(CIFAR10_utils.useModel(image_path=imagePath,
                                    model_path=("./model/top1.pt" if not modelPath else modelPath)))
    input("按任意键返回主菜单...")


def invalid_option():
    os.system('cls')
    print("无效的选项，请重新输入")
    input("按任意键返回主菜单...")


def menu_selection(option):
    if option == '1':
        train_model()
    elif option == '2':
        accuracy_test()
    elif option == '3':
        model_use()
    elif option == '4':
        visualize_training_data()
    elif option == '5':
        exit("已退出")
    else:
        invalid_option()


if __name__ == '__main__':
    while True:
        main_menu()
        user_option = input("请输入选项号码：")
        menu_selection(user_option)
