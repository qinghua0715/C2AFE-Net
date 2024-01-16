import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from gongju.evalution_segmentation import eval_semantic_segmentation
from gongju.dataset import Dataset_train_test
import random

from Network.Transformer.TransFuse.TransFuse import TransFuse_S
# from Transformer.TransFuse.TransFuse_ECA import TransFuse_S

import gongju.cfg as cfg
import numpy as np
import torchvision.utils as vutils
from gongju import calculation_network_model_parameters as tj
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


seed = 1234
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.cuda.manual_seed(seed)


# writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
writer = SummaryWriter(comment='wyl', filename_suffix="wyl")

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

BATCH_SIZE = 12

Cam_test = Dataset_train_test([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.test_crop_size)
test_data = DataLoader(Cam_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


net = TransFuse_S()
net.eval()
net.to(device)
tj.model_structure(net)  # 统计模型参数



# 注册hook
# fmap_dict = {'conv':[]}

# def hook_func(m, i, o):
#     fmap_dict['conv'].append(o)

# net.up1.conv.double_conv[0].register_forward_hook(hook_func)
# # # unet.up2.conv.double_conv[0].register_forward_hook(hook_func)
# # unet.up3.conv.double_conv[0].register_forward_hook(hook_func)
# # net.up4.PyConv3.conv2_3[6].register_forward_hook(hook_func)
# net.inc.double_conv[3].register_forward_hook(hook_func)

# cpu测试
# net.load_state_dict(t.load('./xunlianyanzheng - 防止过拟合/Unet/第一次/6.pth',map_location=t.device('cpu')))
# gpu测试
net.load_state_dict(t.load('./weight/9.pth'))
# net.load_state_dict(t.load('./shujutongji/weight/ISIC/TransFuse/第二次/47.pth'))

# GPU并行训练数据加载到cpu
# state_dict_load = t.load('0.pth', map_location="cpu")
# from collections import OrderedDict
# new_state_dict = OrderedDict()

# for k, v in state_dict_load.items():
# 	namekey = k[7:] if k.startswith('module.') else k
# 	new_state_dict[namekey] = v
# print("new_state_dict:\n{}".format(new_state_dict))
# net.load_state_dict(new_state_dict)

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0
JS = 0
jaccard = 0
DC = 0
SP = 0
SE = 0
PC = 0
RE = 0
RVD = 0
VOE = 0

with t.no_grad():
    print("开始模型测试")
    test_bar = tqdm(test_data, colour='blue')
    for i, sample in enumerate(test_bar):
        data = Variable(sample['img']).to(device)
        label = Variable(sample['label']).to(device)
        _,_,out = net(data)
        out = F.log_softmax(out, dim=1)

        # add image可视化特征图
        # fmap = fmap_dict['conv']
        # fmap = t.stack(fmap)
        # fmap.squeeze_(0)
        # print(fmap.shape)
        #
        # fmap.transpose_(0, 1)
        # print(fmap.shape)
        # print(fmap.type())
        #
        # nrow = int(np.sqrt(fmap.shape[0]))
        # fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
        # writer.add_image('feature map in conv1', fmap_grid, global_step=322)
        #
        # writer.close()

        # 评估
        preout = out.max(dim=1)[1].data.cpu().numpy()
        gtout = label.data.cpu().numpy()

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = label.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrics = eval_semantic_segmentation(pre_label, true_label, preout, gtout)
        train_acc = eval_metrics['mean_class_accuracy'] + train_acc
        train_miou = eval_metrics['miou'] + train_miou

        JS = eval_metrics['JS'] + JS
        DC = eval_metrics['DC'] + DC
        SP = eval_metrics['SP'] + SP
        SE = eval_metrics['SE'] + SE
        PC = eval_metrics['PC'] + PC
        RE = eval_metrics['RE'] + RE
        RVD = eval_metrics['RVD'] + RVD
        VOE = eval_metrics['VOE'] + VOE

        if len(eval_metrics['class_accuracy']) < 2:
            eval_metrics['class_accuracy'] = 0
            train_class_acc = train_class_acc + eval_metrics['class_accuracy']
            error += 1
        else:
            train_class_acc = train_class_acc + eval_metrics['class_accuracy']

        # print(eval_metrics['class_accuracy'], '================', i)
        test_bar.desc = "test iteration[{}/{}] DICE:{:.3f}".format(i + 1,
                                                                   len(test_bar),
                                                                   eval_metrics['miou'])

epoch_str = (
    'JS :{:.5f}, DC :{:.5f}, SP :{:.5f}, SE :{:.5f}, PC :{:.5f}, RE :{:.5f}, RVD :{:.5f}, VOE:{:.5f}, test_acc :{:.5f} ,test_miou :{:.5f}, '
    'test_class_acc :{:}'.format(
        JS / (len(test_data) - error),
        DC / (len(test_data) - error),
        SP / (len(test_data) - error),
        SE / (len(test_data) - error),
        PC / (len(test_data) - error),
        RE / (len(test_data) - error),
        RVD / (len(test_data) - error),
        VOE / (len(test_data) - error),
        train_acc / (len(test_data) - error),
        train_miou / (len(test_data) - error),
        train_class_acc / (len(test_data) - error)

    ))

print(epoch_str + '==========last')
