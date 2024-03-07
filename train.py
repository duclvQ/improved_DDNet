#!/usr/bin/env python
# coding: utf-8



# In[2]:


from pathlib import Path
import matplotlib.pyplot as plt
from torch import log
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

from dataloader.cobot_loader import load_cobot_data, Cdata_generator, CConfig

from models.DDNet_Original import DDNet_Original as DDNet
from utils import makedir
import sys
import time
import numpy as np
import logging

import clip
from KLLoss import KLLoss, CELoss
import warnings
import argparse
warnings.filterwarnings("ignore")
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# In[ ]:

parser = argparse.ArgumentParser(description='_')
parser.add_argument('--run_name', type=str, default='RN50')
run_name = parser.parse_args().run_name



# In[3]:



# In[4]:

# description of actions

prefix = "A person "
list_of_texts = [
f'{prefix} does nothing',
f'{prefix}Raise one arm up high, make a fist, then keep the fist still, bend the arm to shoulder level, then raise it up again.,'

f'{prefix}Place one hand in front of your chest. Then move the arm in the opposite direction.',

f'{prefix}Place your arms vertical and perpendicular to the ground at shoulder level when your arms are bent. Then bring it back to your chest.',

f'{prefix}Raise one hand at chest level, palm facing down, tip of the hand facing out, then press the hand down to hip level, then lift the arm up and bring it back to the original position.',

f'{prefix}Place one hand flat on your hip, tip of the hand facing outward. Then, lift the hand lying face up to chest level, then press the hand down again and return to the starting position.',

f'{prefix}Place one arm across your chest, palm facing down, tip of the hand facing the opposite hand. Then, move your arm along the chest plane at an angle of more than 120 degrees, at which time the tip of your hand points outward. Finally return your arms to the starting position.',

f'{prefix}Raise one arm overhead, holding the hand, then move the arm and keep the holding hand towards the top of the head. Finally, return your arms to the starting position.',

f'{prefix}Raise one arm horizontally across the chest, vertical and perpendicular to the ground, hand with one index finger pointing up to the sky. Then, move your arm and index finger up until your arm is at shoulder level. Finally, return to the original position.',

f'{prefix}Raise one arm, place the hand on one half of the head.',

f'{prefix}Place one arm across your chest, vertical to your body, perpendicular to the ground. Raise your index finger and thumb normally, with your index finger pointing toward the sky.',

f'{prefix}Place your arms in front of your chest, clasp your hands, spread out only your thumbs, stand up and face the sky. Hold your hands facing in front of your body.',

f'{prefix}Place your arms in front of your chest, hands open and facing each other, palm tips facing the sky.',

f'{prefix}Place your arms across your chest, your hands close together (about a few centimeters), your palms facing each other.',

f'{prefix}Place one arm in front of your chest, vertical to the body. Palms face out in front of the body, fingers clustered in a claw shape.',

f'{prefix}Place one arm in front of your chest, vertical to the body. Hands clenched.',

f'{prefix}Place your hands across your chest, arms pointing out in front of your body. The hand bends the ring finger and little finger, the middle finger contacts the thumb, and the index finger spreads out naturally.',

f'{prefix}Place one arm across the neck, the arm horizontally perpendicular to the body axis, the tip of the hand facing the opposite shoulder. The other arm places the hand in front of the chest perpendicular to the palm of the other hand, the tip of the hand facing up.',

f'{prefix}Place one hand on the opposite shoulder, arms spread out with five fingers embracing one shoulder.',

f'{prefix}Place one hand on the opposite shoulder, placing the hand close to the shoulder. Hold the 3 thumbs, ring finger and little finger, and spread the index and middle fingers.',

f'{prefix}Raise one arm in front of your chest, arm vertical to the body axis. The fingers spread out and come together at one point, the tips of the hands facing the front of the body.']
# Now, you have a list of detailed action descriptions.
len(list_of_texts)


# In[5]:
# reordering the list of texts

list_of_labels  = ['START',
'STOP',
'SLOWER',
'FASTER',
'DONE',
'FOLLOW ME',
'LIFT',
'HOME',
'LOOK',
'OK',
'HELP',
'AGAIN',
'PICKPART',
'DEPOSIT PART',
'INTERACTION',
'JOYSTICK',
'IDENTIFICATION',
'CHANGE',
'REPORT',]


a = ['No gesture','Start', 'Stop', 'Slower', 'Faster', 'Done', 'FollowMe', 'Lift', 'Home', 'Interaction', 'Look', 'PickPart', 'DepositPart', 'Report', 'Ok', 'Again', 'Help', 'Joystick', 'Identification', 'Change']
b = ['No gesture','START', 'STOP', 'SLOWER', 'FASTER', 'DONE', 'FOLLOW ME', 'LIFT', 'HOME', 'LOOK', 'OK', 'HELP', 'AGAIN', 'PICKPART', 'DEPOSIT PART', 'INTERACTION', 'JOYSTICK', 'IDENTIFICATION', 'CHANGE', 'REPORT']

# Convert both lists to lowercase for comparison
a_lower = [item.lower() for item in a]
b_lower = [item.lower().replace(' ', '') for item in b]

# Get the order of items in list b based on list a
# Create a dictionary with the order of items in list a
order = {item: i for i, item in enumerate(a_lower)}

# Sort list b based on the order in list a
list_of_texts_sorted = sorted(list_of_texts, key=lambda item: order.get(item.lower().replace(' ', ''), float('inf')))

print(list_of_texts_sorted)


# In[6]:

# set device, always use cuda
device = "cuda:0"

def text_prompt_openai_random():
    return clip.tokenize(list_of_texts_sorted)


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model.float()

    def forward(self,text):
        return self.model.encode_text(text)

token_list = torch.tensor(np.array(text_prompt_openai_random()))
token_list = token_list.to(device)


# In[7]:

# set up logging, and result saving
sys.path.insert(0, './pytorch-summary/torchsummary/')
from torchsummary import summary  # noqa

savedir = Path('experiments') / Path(str(int(time.time())) + '_' + run_name)
makedir(savedir)
logging.basicConfig(filename=savedir/'train.log', level=logging.INFO)
history = {
    "train_loss": [],
    "test_loss": [],
    "test_acc": [],
    "text_loss":[]
}
#
args = None
save_model = True
log_interval = 100


# In[8]:


def create_logits(x1, x2, logit_scale=10):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt


# In[9]:
model_, preprocess = clip.load(run_name, device)
model_text = TextCLIP(model_)
model_text = model_text.cuda(device)
token_list = np.array(text_prompt_openai_random())
token_list = torch.tensor(token_list).to(device)    
text_embedding_list = model_text(token_list).float()

def train(args, model, text_model, token_list, device, train_loader, optimizer, epoch, criterion, KLloss):
    model.train()

    #token_list = token_list.to(device)
    train_loss = 0
    text_loss_mean = 0
    correct = 0
    for batch_idx, (data1, data2, target) in enumerate(tqdm(train_loader)):
        M, P, target = data1.to(device), data2.to(device), target.to(device)
        

        optimizer.zero_grad()
        output, _features = model(M, P)
        # get target token
        label_g = gen_label(target)
        
        text_embedding = list()
        for i in  range(target.size(0)):
            text_embedding.append(text_embedding_list[target[i]])
        # convert to tensor
        text_embedding = torch.stack(text_embedding)
        text_embedding = text_embedding.to(device)
        #print(_features.size())
        logits_per_image, logits_per_text = create_logits(_features,text_embedding, 50)
        ground_truth = torch.tensor(label_g,dtype=text_embedding.dtype,device=device)
        loss_imgs = KLloss(logits_per_image,ground_truth)
        loss_texts = KLloss(logits_per_text, ground_truth)
        text_loss_mean += (loss_texts).detach().item()
        # clear memory of text_embedding
        del text_embedding
        # Calculate train accuracy
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        #print("correct:", correct)
        # using KL loss when training
        loss = criterion(output, target) + 0.1*(loss_imgs + loss_texts)/ 2
        # the below loss is CE loss, used for original DDNet, uncomment it if you want to use it
        #loss = criterion(output, target) 
        
        train_loss += loss.detach().item()
        loss.backward(retain_graph=True)
        optimizer.step()
        # Calculate accuracy
        accuracy = (predicted == target).sum().item() / target.size(0) * 100

        if batch_idx % log_interval == 0:
            msg = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), accuracy))
            print(msg)
            logging.info(msg)
            dry_run = False
            if dry_run:
                break
    print("mean text_img_loss:", text_loss_mean/(batch_idx))
    text_loss = text_loss_mean/(batch_idx)
    history['train_loss'].append(train_loss)
    history['text_loss'].append(text_loss)
    return train_loss


# save acc test to txt file
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for _, (data1, data2, target) in enumerate(tqdm(test_loader)):
            M, P, target = data1.to(device), data2.to(device), target.to(device)
            output, _features = model(M, P)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # output shape (B,Class)
            # target_shape (B)
            # pred shape (B,1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(correct / len(test_loader.dataset))
    msg = ('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    with open(savedir / 'test.txt', 'a') as f:
        f.write(msg)
    
    print(msg)
    logging.info(msg)


# In[10]:


device = torch.device("cuda")
kwargs = {'batch_size': 16}
kwargs.update({'num_workers': 12,
                    'pin_memory': True,
                    'shuffle': True},)

Config = CConfig()
data_generator = Cdata_generator
load_data = load_cobot_data
clc_num = Config.clc_num


# 

# In[11]:


C = Config
Train, Test, le = load_cobot_data()
X_0, X_1, Y = data_generator(Train, C, le)
X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')
X_1 = torch.from_numpy(X_1).type('torch.FloatTensor')
Y = torch.from_numpy(Y).type('torch.LongTensor')

X_0_t, X_1_t, Y_t = data_generator(Test, C, le)
X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
X_1_t = torch.from_numpy(X_1_t).type('torch.FloatTensor')
Y_t = torch.from_numpy(Y_t).type('torch.LongTensor')

trainset = torch.utils.data.TensorDataset(X_0, X_1, Y)
train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=16)


# In[12]:
# if main to run this file
if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=UserWarning)
    # add argument
    
   

    Net = DDNet(C.frame_l, C.joint_n, C.joint_d,
                C.feat_d, 64, clc_num, run_name)
    model = Net.to(device)

    


    #summary(model, [(C.frame_l, C.feat_d), (C.frame_l, C.joint_n, C.joint_d)])
    #optimizer = optim.Adam(list(model.parameters() ), lr=0.01, betas=(0.9, 0.999))
    optimizer = optim.SGD(list(model.parameters()) , lr=0.01, momentum=0.9)

    #optimizer = optim.Adam(list(model.parameters()), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    KLloss = KLLoss().cuda("cuda:0")
    from torch.optim.lr_scheduler import MultiStepLR


    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, cooldown=0.5, min_lr=5e-6, verbose=True)
    #scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    # save that best acc model
    # if test.txt exists, remove it
    if (savedir / 'test.txt').exists():
        os.remove(savedir / 'test.txt')
    for epoch in range(1,  400):
        train_loss = train(args, model, model_text, token_list, device, train_loader, optimizer, epoch, criterion, KLloss)
        test_loss = test(model, device, test_loader)
        scheduler.step(train_loss)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # Now you can use ax1, ax2, ax3
    
    ax1.legend(['Train', 'Test'], loc='upper left')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Loss')

    ax2.set_title('Model accuracy') 
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.plot(history['test_acc'])
    xmax = np.argmax(history['test_acc'])
    ymax = np.max(history['test_acc']) 
    text = "x={}, y={:.3f}".format(xmax, ymax)
    ax2.annotate(text, xy=(xmax, ymax))

    ax3.set_title('Confusion matrix')
    model.eval()
    with torch.no_grad():
        Y_pred, _ = model(X_0_t.to(device), X_1_t.to(device))  # Unpack the tuple
        Y_pred = Y_pred.cpu().numpy()  # Now you can call .cpu() on Y_pred
    Y_test = Y_t.numpy()
    cnf_matrix = confusion_matrix(Y_test, np.argmax(Y_pred, axis=1))
    ax3.imshow(cnf_matrix)
    fig.tight_layout()
    plt.savefig(str(savedir/'my_plot.png'))
    if save_model:
        torch.save(model.state_dict(), str(savedir/"model.pt"))

    # draw text_loss and test_acc
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Text-Skel Loss')
    ax1.set_ylabel('Text-Skel Loss')
    ax1.set_xlabel('Epoch')
    ax1.plot(history['text_loss'])
    ax2.set_title('Model accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.plot(history['test_acc'])
    xmax = np.argmax(history['test_acc'])   
    ymax = np.max(history['test_acc'])
    text = "x={}, y={:.3f}".format(xmax, ymax)
    ax2.annotate(text, xy=(xmax, ymax))
    fig.tight_layout()
    plt.savefig(str(savedir/'text_loss.png'))
    
    

    if False:
        device = ['cpu', 'cuda']
        # calc time
        for d in device:
            tmp_X_0_t = X_0_t.to(d)
            tmp_X_1_t = X_1_t.to(d)
            model = model.to(d)
            # warm up
            _ = model(tmp_X_0_t, tmp_X_1_t)

            tmp_X_0_t = tmp_X_0_t.unsqueeze(1)
            tmp_X_1_t = tmp_X_1_t.unsqueeze(1)
            start = time.perf_counter_ns()
            for i in range(tmp_X_0_t.shape[0]):
                _ = model(tmp_X_0_t[i, :, :, :], tmp_X_1_t[i, :, :, :])
            end = time.perf_counter_ns()
            msg = ("total {}ns, {:.2f}ns per one on {}".format((end - start),
                                                                ((end - start) / (X_0_t.shape[0])), d))
            print(msg)
            logging.info(msg)
