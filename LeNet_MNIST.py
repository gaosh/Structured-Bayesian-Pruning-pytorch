import torchvision.datasets as dset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from LeNet import LeNet_SBP, LeNet, accuracy
import torch.optim as optim
from torch.autograd import Variable
import os

batchsize = 128
epoch = 60
learning_rate = 0.001
alpha = 0.01
path = './SBP_model'
pretrain = False

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,)),
])

lenet = LeNet()
lenet.cuda()

train_set = dset.MNIST(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize,
                                           shuffle=True, num_workers=2)

test_set = dset.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchsize,
                                          shuffle=False, num_workers=2)


params = [
    {'params': lenet.parameters()},
 ]
optimizer = optim.Adam(params, lr=learning_rate, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)

num_batch = len(train_loader)
print(num_batch)
criterion = nn.CrossEntropyLoss()
best_result = 0

if pretrain:
    for e in range(epoch):
        lenet.train()
        running_loss = 0.0
        running_klloss = 0.0
        for x_batch, y_batch in train_loader:

            x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
            prediction = lenet(x_batch)
            lambda_0 = 1.0
            loss = criterion(prediction, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.data[0]
            running_loss += batch_loss

        lenet.eval()
        train_accuracy = accuracy(train_loader, lenet)
        test_accuracy = accuracy(test_loader, lenet)
        lenet.train()
        if(test_accuracy>=best_result):
            best_result = test_accuracy
            torch.save(lenet.state_dict(), os.path.join(path, 'lenet_best.pt'))
        print('Epoch [%d], Loss: %.4f, KL: %.4f, Train accuracy: %.4f, Test accuracy: %.4f, Best: %.4f' % (e, running_loss/num_batch, running_klloss/num_batch,train_accuracy, test_accuracy, best_result))




alex_path = os.path.join(path, "lenet_best.pt")
lenet_best = LeNet()
lenet_best.load_state_dict(torch.load(alex_path))

lenet_sbp = LeNet_SBP()
sbp_learningrate = 2e-5
sbp_parameters = [
    {'params': lenet_sbp.conv1.weight},
    {'params': lenet_sbp.conv2.weight},
    {'params': lenet_sbp.fc1.weight},
    {'params': lenet_sbp.fc2.weight},

    {'params': lenet_sbp.sbp_1.log_sigma, 'lr': 10*sbp_learningrate},
    {'params': lenet_sbp.sbp_2.log_sigma, 'lr': 10*sbp_learningrate},
    {'params': lenet_sbp.sbp_3.log_sigma, 'lr': 10*sbp_learningrate},
    {'params': lenet_sbp.sbp_4.log_sigma, 'lr': 10*sbp_learningrate},

    {'params': lenet_sbp.sbp_1.mu, 'lr': 10*sbp_learningrate},
    {'params': lenet_sbp.sbp_2.mu, 'lr': 10*sbp_learningrate},
    {'params': lenet_sbp.sbp_3.mu, 'lr': 10*sbp_learningrate},
    {'params': lenet_sbp.sbp_4.mu, 'lr': 10*sbp_learningrate},

 ]

sbp_optimizer = optim.Adam(sbp_parameters, lr=sbp_learningrate, betas=[0.95,0.999])
sbp_scheduler = optim.lr_scheduler.StepLR(sbp_optimizer, step_size= 250,gamma=0.1)
finetune_epoch = 300

lenet_sbp.conv1.weight = lenet_best.conv1.weight
lenet_sbp.conv2.weight = lenet_best.conv2.weight
lenet_sbp.fc1.weight = lenet_best.fc1.weight
lenet_sbp.fc2.weight = lenet_best.fc2.weight
lenet_sbp.cuda()

for e in range(finetune_epoch):
    lenet_sbp.train()
    running_loss = 0.0
    running_klloss = 0.0
    for x_batch, y_batch in train_loader:

        x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
        prediction,kl_loss = lenet_sbp(x_batch)

        loss = criterion(prediction, y_batch) +  kl_loss

        sbp_optimizer.zero_grad()
        loss.backward()
        sbp_optimizer.step()

        batch_loss = loss.data[0]
        running_loss += batch_loss
        running_klloss+=kl_loss

    lenet_sbp.eval()
    train_accuracy = accuracy(train_loader, lenet_sbp)
    test_accuracy = accuracy(test_loader, lenet_sbp)
    lenet_sbp.train()
    if(test_accuracy>=best_result):
        best_result = test_accuracy


    print('Epoch [%d], Loss: %.4f, KL: %.4f, Train accuracy: %.4f, Test accuracy: %.4f, Best: %.4f' % (e, running_loss/num_batch, running_klloss/num_batch,train_accuracy, test_accuracy, best_result))
    if (e + 1)% 5 == 0:
        sparsity_arr = lenet_sbp.layerwise_sparsity()
        print('l1-Sparsity: %.4f, l2-Sparsity: %.4f, l3-Sparsity: %.4f, l4-Sparsity: %.4f' %(sparsity_arr[0], sparsity_arr[1], sparsity_arr[2],sparsity_arr[3]))
        snr_arr = lenet_sbp.display_snr()
        print('l1-snr: %.4f, l2-snr: %.4f, l3-snr: %.4f, l4-snr: %.4f' % (
            snr_arr[0], snr_arr[1], snr_arr[2], snr_arr[3]))
