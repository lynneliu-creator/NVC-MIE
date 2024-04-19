from __future__ import print_function
import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10,CIFAR100
from models.resnet_new import ResNet18,ResNet34
import numpy as np
from torch.autograd import Variable
from TinyImageNet import TinyImageNetDataset
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#from FWA.frank_wolfe import FrankWolfe
from autoattack import AutoAttack
from advertorch.attacks import CarliniWagnerL2Attack, DDNL2Attack, SpatialTransformAttack
#from utils_attack.FWA import LinfFWA
#from utils_attack.ti_dim_gpu import TIDIM_Attack

parser = argparse.ArgumentParser(description='PyTorch  Adversarial Training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--data-dir', default='./data/cifar10', 
                    help='path to dataset (default: ../MI_project/data/cifar10)')
parser.add_argument('--epsilon', default=0.031, help='perturbation')#0.031
parser.add_argument('--num-steps', default=40, help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, help='perturb step size') # epsilon/4
#'./checkpoint/cifar10/MIAT_copula_linf',#'./checkpoint/cifar100/copula/MIAT_l2',
parser.add_argument('--model-dir', default='./checkpoint/resnet/MIAT_copula_l2',
                    help='directory of model for saving checkpoint')
parser.add_argument('--print_freq', type=int, default=1)


args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def craft_adversarial_example_pgd(model, x_natural, y, step_size=0.125,
                epsilon=0.5, perturb_steps=10, distance='l_2'):

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_kl = F.cross_entropy(logits, y.long())

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        batch_size = len(x_natural)
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.cross_entropy(model(adv), y)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)

    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def craft_adversarial_example(model, x_natural, y, step_size=0.125,
                epsilon= 0.5, perturb_steps=10, distance='l_2'):

     
    adversary = AutoAttack(model, norm='Linf', eps=8 / 255, version='standard')
    '''
    mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float, device='cuda').unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float, device='cuda').unsqueeze(-1).unsqueeze(-1)
    unnormalize = lambda x: x * std + mu
    normalize = lambda x: (x - mu) / std
    #linf FWA
    attacker=FrankWolfe(predict=model,
                        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                        eps=0.01,
                        kernel_size=4,
                        nb_iter=40,
                        entrp_gamma=1e-3,
                        dual_max_iter=50,
                        grad_tol=1e-4,
                        int_tol=1e-4,
                        device='cuda',
                        postprocess=False,
                        verbose=False)

    
    '''
    '''
    adversary = DDNL2Attack(model, nb_iter=20, gamma=0.05, init_norm=1.0, quantize=True, levels=16, clip_min=0.0,
                            clip_max=1.0, targeted=False, loss_fn=None)
    
    '''
    '''
    adversary = CarliniWagnerL2Attack(
        model, 100, clip_min=0.0, clip_max=1.0, max_iterations=10, confidence=1, initial_const=1, learning_rate=1e-2,
        binary_search_steps=4, targeted=False)
    '''


    '''
    adversary = LinfFWA(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                      eps=8/255, kernel_size=4, lr=0.007, nb_iter=40, dual_max_iter=15, grad_tol=1e-4,
                    int_tol=1e-4, device="cuda", postprocess=False, verbose=True)
    

    '''
    '''
    adversary = SpatialTransformAttack(
            model, 10, clip_min=0.0, clip_max=1.0, max_iterations=10, search_steps=5, targeted=False)
    '''

    '''
    adversary = TIDIM_Attack(model,
                       decay_factor=1, prob=0.5,
                       epsilon=8/255, steps=40, step_size=0.01,
                       image_resize=33,
                       random_start=False)
    '''


    '''
    adversary = TIDIM_Attack(eps=8/255, steps=40, step_size=0.007, momentum=0.1, prob=0.5, clip_min=0.0, clip_max=1.0,
                 device=torch.device('cuda'), low=32, high=32)
    '''
    #autoattack
    x_adv = adversary.run_standard_evaluation(x_natural, y, bs=args.batch_size)
    #advertorch
    #x_adv = adversary.perturb(x_natural, y)
    #
    #x_adv = adversary.perturb(model, x_natural, y)
    #FWA
    #x_adv=attacker.perturb(x_natural,y)
    return x_adv


def evaluate_at(i, data, label, model):
    model.eval()  # Change model to 'eval' mode.

    data_adv = craft_adversarial_example_pgd(model=model, x_natural=data, y=label, step_size=args.step_size,
                                         epsilon=args.epsilon, perturb_steps=40, distance='l_inf')

    logits = model(data)
    outputs = F.softmax(logits, dim=1)

    _, pred = torch.max(outputs.data, 1)

    correct1 = (pred == label).sum()


    logits = model(data_adv)
    outputs = F.softmax(logits, dim=1)

    _, pred = torch.max(outputs.data, 1)

    correct_adv = (pred == label).sum()

    acc1 = 100 * float(correct1) / data.size(0)

    acc_adv = 100 * float(correct_adv) / data.size(0)

    if (i + 1) % args.print_freq == 0:
        print(
            'Iter [%d/%d] Test classifier: Nat Acc: %.4f; Adv Acc: %.4f'
            % (i + 1, 10000 // data.size(0), acc1, acc_adv))

    return [acc1, acc_adv]



def main(args):
    # settings
    setup_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model_dir = args.model_dir

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    # setup data loader
    trans = transforms.Compose([
        transforms.ToTensor()
    ])



    #testset = TinyImageNetDataset(root="./", mode='val',download=False,transform=trans)
    testset = CIFAR10(args.data_dir,train=False,download=False,transform=trans)
    test_loader = DataLoader(testset,batch_size=args.batch_size,shuffle=True,drop_last=True,pin_memory=torch.cuda.is_available())
    print("测试样本个数：",len(testset)) 


    # classifier = resnet.ResNet18(10)
    classifier = ResNet18(10)
    # classifier = VGGNet19()
    classifier.load_state_dict(torch.load(os.path.join(model_dir, 'target_model.pth')))
    classifier = classifier.to(device)

    cudnn.benchmark = True

    # Adversraial test
    print('------Starting test------')

    total_num = 0.
    test_nat_correct = 0.
    test_adv_correct = 0.
    i=0


    for i, (data, labels) in enumerate(test_loader):
        data = data.cuda()
        labels = labels.cuda()

        classifier.eval()
        test_acc = evaluate_at(i, data, labels, classifier)
        i+=1

        total_num += 1
        test_nat_correct += test_acc[0]
        test_adv_correct += test_acc[1]

    test_nat_correct = test_nat_correct / total_num
    test_adv_correct = test_adv_correct / total_num

    print('Test Classifer: Nat ACC: %.4f; Adv ACC: %.4f' % (test_nat_correct, test_adv_correct))


if __name__ == '__main__':
    main(args)
