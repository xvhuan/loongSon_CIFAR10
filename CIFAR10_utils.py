import sys
import time

import torch.nn as nn
import torch.utils
import torch.utils
import torchvision.datasets as dset
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import utils
from model import NetworkCIFAR as Network
from nasnet_set import *
from utils import *


class Cifar10:
    def __init__(self):
        # 判断有无文件夹，没有则创建
        if not os.path.exists("model"):
            os.mkdir("model")

        np.random.seed(88)
        torch.manual_seed(88)
        self.writer = SummaryWriter()
        self.model_path = 'model/top1.pt'
        self.device = torch.device("cuda:0")
        self.ds_train = None
        self.ds_test = None

    def loader(self):
        # 对数据集的预处理
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        train_transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), AutoAugment()]
        train_transform.extend([transforms.ToTensor(),
                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD), ])
        train_transform = transforms.Compose(train_transform)
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        self.ds_train = dset.CIFAR10(root="./data", train=True, download=True, transform=train_transform)

        self.ds_test = dset.CIFAR10(root="./data", train=False, download=True, transform=valid_transform)

    def trainModel(self, num_epochs, lr, batch_size):
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        self.loader()
        # 将测试集和训练集进行提取并乱序（每个类别选取5000张图片）
        train_indices = []
        test_indices = []
        for class_label in range(10):
            indices_train = np.where(np.array(self.ds_train.targets) == class_label)[0]
            indices_test = np.where(np.array(self.ds_test.targets) == class_label)[0]
            np.random.shuffle(indices_train)
            train_indices.extend(indices_train[:5000])  # 从每个类别中随机抽取5000个样本加入训练集
            np.random.shuffle(indices_test)
            test_indices.extend(indices_test[:1000])  # 从每个类别中随机抽取1000个样本加入测试集
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        trainLoader = torch.utils.data.DataLoader(
            CutMix(self.ds_train, 10,
                   beta=1.0, prob=0.5, num_mix=2),
            batch_size=batch_size, num_workers=2, pin_memory=True, sampler=train_sampler)

        testLoader = torch.utils.data.DataLoader(self.ds_test, batch_size=batch_size, sampler=test_sampler,
                                                 num_workers=0)
        continue_train = False
        if os.path.exists("model/model.pt"):
            continue_train = True
        cur_epoch = 0
        net = eval('[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]')
        code = gen_code_from_list(net, node_num=int((len(net) / 4)))
        genotype = translator([code, code], max_node=int((len(net) / 4)))

        if not continue_train:
            model = Network(128, 10, 24, True, genotype).to(self.device)
            criterion = CutMixCrossEntropyLoss(True).to(self.device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr,
                momentum=0.9,
                weight_decay=3e-4
            )
            model_ema = ModelEma(
                model,
                decay=0.9999,
                device='cpu')
        else:
            print('continue train from checkpoint')
            model = Network(128, 10, 24, True, genotype).to(self.device)
            criterion = CutMixCrossEntropyLoss(True).to(self.device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr,
                momentum=0.9,
                weight_decay=3e-4
            )
            checkpoint = torch.load(self.model_path,self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            cur_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model_ema = ModelEma(
                model,
                decay=0.9999,
                device='cpu',
                resume=self.model_path)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

        best_acc = 0.0

        if continue_train:
            for i in range(cur_epoch + 1):
                scheduler.step()

        for epoch in range(cur_epoch, num_epochs):
            print('cur_epoch is', epoch)
            logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
            model.drop_path_prob = 0.2 * epoch / 600
            model_ema.ema.drop_path_prob = 0.2 * epoch / 600

            train(trainLoader, model, criterion, optimizer, epoch, model_ema)
            scheduler.step()
            valid_acc_ema, valid_obj_ema = infer(testLoader, model_ema.ema, criterion)
            logging.info('valid_acc_ema %f', valid_acc_ema)
            self.writer.add_scalar('Validation_EMA/Loss', valid_obj_ema, epoch)
            self.writer.add_scalar('Validation_EMA/Accuracy', valid_acc_ema, epoch)

            valid_acc, valid_obj = infer(testLoader, model, criterion)
            logging.info('valid_acc: %f', valid_acc)
            self.writer.add_scalar('Validation/Loss', valid_obj, epoch)
            self.writer.add_scalar('Validation/Accuracy', valid_acc, epoch)

            if valid_acc > best_acc:
                best_acc = valid_acc
                print('this model is the best')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, os.path.join(self.model_path))
            print('current best acc is', best_acc)
            logging.info('best_acc: %f', best_acc)

            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'state_dict_ema': get_state_dict(model_ema)},
                os.path.join("model/model.pt"))

    def testModel(self):
        self.loader()
        # 异常处理
        try:
            print(f"模型位置：{self.model_path}")
            print("验证各类别准确率...")
            net = eval('[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]')
            code = gen_code_from_list(net, node_num=int((len(net) / 4)))
            genotype = translator([code, code], max_node=int((len(net) / 4)))
            model = Network(128, 10, 24, True, genotype).to(self.device)

            checkpoint = torch.load('./model/top1.pt',map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            criterion = nn.CrossEntropyLoss().to(self.device)

            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])

            valid_queue = torch.utils.data.DataLoader(
                dset.CIFAR10(root="./data", train=False, transform=valid_transform),
                batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

            model.eval()
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            class_correct = [0.0] * 10
            class_total = [0.0] * 10
            correct = 0
            total = 0

            with torch.no_grad(), tqdm(total=len(valid_queue)) as pbar:
                for step, (x, target) in enumerate(valid_queue):
                    x = x.cuda()
                    target = target.cuda(non_blocking=True)

                    logits, _ = model(x)
                    loss = criterion(logits, target)
                    pred = logits.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

                    # Compute class-wise accuracy
                    c = pred.eq(target)
                    for i in range(len(target)):
                        label = target[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

                    pbar.update(1)
                    pbar.set_description(f'Validation: Loss={loss.item():.3f} | Acc={100.0 * correct / total:.3f}%')
            valid_acc = 100.0 * correct / total

            for i in range(10):
                print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
            print('Final Validation Accuracy: {:.2f}%'.format(valid_acc))
        except FileNotFoundError:
            print("error：未找到模型文件，请检查model文件夹下存在top1.pth")


def train(train_queue, model, criterion, optimizer, epoch, model_ema):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()

    for step, (x, target) in enumerate(train_queue):

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, target)
        loss_aux = criterion(logits_aux, target)
        loss += 0.4 * loss_aux

        losses.update(loss.item(), x.size(0))

        if len(target.size()) == 1:
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        model_ema.update(model)

        if step % 50 == 0:
            logging.info('train %03d', step)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, ema=False):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (x, target) in enumerate(valid_queue):
        x = x.to(torch.device("cpu"))
        target = target.to(torch.device("cpu"))

        with torch.no_grad():
            logits, _ = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if step % 50 == 0:
            if not ema:
                logging.info('>>Validation: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            else:
                logging.info('>>Validation_ema: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def useModel(image_path, model_path="model/top1.pt"):
    def predict_image(image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 修改大小的方式
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        image = transform(image).unsqueeze(0).to(torch.device('cpu'))

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output[0], 1)
            species = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            species_china = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
            predicted_label = species_china[predicted.item()]
        return predicted_label

    # 加载已训练的模型
    net = eval('[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]')
    code = gen_code_from_list(net, node_num=int((len(net) / 4)))
    genotype = translator([code, code], max_node=int((len(net) / 4)))
    model = Network(128, 10, 24, True, genotype).to(torch.device('cpu'))
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置模型为评估模式

    startTime = time.time()
    try:
        predicted_label = predict_image(image_path)
    except Exception:
        raise Exception("图片路径错误")
    endTime = time.time()
    d = {"result": predicted_label, "time": endTime - startTime}
    return d
