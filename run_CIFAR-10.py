import argparse
import heapq
import math
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.vgg import vgg13
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from settings.config import parse_argument
from settings.datasetting import get_dataset
import copy
# from models.cnn import CNN
import os
import json



class Train_info:
    def __init__(self,time,node_id):
        self.time = time
        self.node_id = node_id

    def __lt__(self,train_info):
        return self.time < train_info.time
    
class Train:
    def __init__(self,args):
        np.random.seed(seed=1333)
        self.training_method = args.training_method
        self.dataset = args.dataset
        self.cnn = args.cnn
        self.model = None
        self.model_zero = None
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.lr = 0.01
        # 訓練時のバッチサイズ
        self.batchsize = args.batchsize
        # テスト時のバッチサイズ
        self.test_batchsize = args.test_batchsize
        # ノード数
        self.worker_num = args.worker_num
        # 何イテレータごとに送信するか
        self.T = 1
        # 遅延 [秒]
        self.latency = 0.005
        # モデルの容量 [MiB]
        self.model_size = 54
        # 広い帯域幅は狭い帯域幅の何倍か（狭いのを1 Gbpsとする）
        self.wide_bandwidth = 10
        # 帯域幅が広いノード数
        self.wide_node_size = 10
        # 1イテレーションにかかる時間
        self.iter_time = 0.102
        # 計算時間の計測をするか
        self.time_flag = False
        # Non-iid環境
        self.Non_iid = args.non_iid
        self.alpha_size=args.alpha_size
        self.alpha_label=args.alpha_label




    def learn_model(self, model, train_loader, optimizer, lr=0.1):
        if self.time_flag:
            torch.cuda.synchronize()
            start = time.perf_counter()

        # ---------- One iteration of the training loop ----------
        for image_train, target_train in train_loader:
            image_train, target_train = image_train.to(self.device), target_train.to(self.device)
            break

        optimizer = optim.SGD(model.parameters(),lr=lr)

        prediction = model(image_train)
        loss = F.cross_entropy(prediction,target_train)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # あらかじめ設定された固定時間にするか否か
        if self.time_flag:
            torch.cuda.synchronize()
            time_diff = time.perf_counter()-start
            return time_diff
        return self.iter_time
    
    def learnMutual_model(self,models,node_id,model,neighbormondel,train_loader,optimizer,lr=0.1):
        if self.time_flag:
            torch.cuda.synchronize()
            start=time.perf_counter()

        optimizer1 = optim.SGD(model.parameters(),lr=lr)
        optimizer2 = optim.SGD(neighbormondel.parameters(),lr=lr)

        for image,label in train_loader:
            break

        y_1=model(image)
        y_2= neighbormondel(image)

        loss_1 = F.cross_entropy(y_1,label)+self.kl_divergence(y_1,y_2.detach())
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        y_1 = model(image)
        loss_2 = F.cross_entropy(y_2,label)+self.kl_divergence(y_2,y_1.detach())
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        # modelの書き換え
        models[node_id] = copy.deepcopy(neighbormondel)

        if self.time_flag:
            torch.cuda.synchronize()
            time_diff = time.perf_counter()-start
            return time_diff
        else:
            return self.iter_time*2.0
    
    def learnEnsamble_model(self,models,node_id,model,neighbormondel,train_loader,optimizer,lr=0.1):
        if self.time_flag:
            torch.cuda.synchronize()
            start=time.perf_counter()

        optimizer1 = optim.SGD(model.parameters(),lr=lr)
        optimizer2 = optim.SGD(neighbormondel.parameters(),lr=lr)

        for image,label in train_loader:
            break

        y_1=model(image)
        y_2= neighbormondel(image)

        loss_1 = F.cross_entropy(y_1,label)+self.kl_ensamble_divergence(y_1,y_2.detach())

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        y_1 = model(image)
        loss_2 = F.cross_entropy(y_2,label)+self.kl_ensamble_divergence(y_2,y_1.detach())
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        models[node_id] = copy.deepcopy(neighbormondel)

        if self.time_flag:
            torch.cuda.synchronize()
            time_diff = time.perf_counter()-start
            return time_diff
        else:
            return self.iter_time*2.0
    
    def kl_divergence(self,logits_1,logits_2):
        softmax_1 = F.softmax(logits_1, dim=1)
        softmax_2 = F.softmax(logits_2, dim=1)
        kl = (softmax_2 * torch.log((softmax_2 / (softmax_1+1e-10)) + 1e-10)).sum(dim=1)
        return kl.mean()
    
    def kl_ensamble_divergence(self,logits_1,logits_2):
        softmax_1 = F.softmax(logits_1, dim=1)
        softmax_2 = F.softmax(logits_2, dim=1)
        ensamble_teacher = (softmax_1+softmax_2)/2
        kl = (ensamble_teacher * torch.log((ensamble_teacher / (softmax_1+1e-10)) + 1e-10)).sum(dim=1)
        return kl.mean()


    def test_models_max(self,models,test_loader,start,end):
        test_accuracies = []
        # modelsは各ノードのモデルが格納されている配列

        for j in range(start,end):
            model = models[j].to(self.device)
            model.eval()
            test_accuracy = 0.0
            with torch.no_grad():
                for image_test,target_test in test_loader:
                
                    image_test = image_test.to(self.device)

                    prediction_test = model(image_test)
        
                    test_accuracy += (prediction_test.max(1)[1]==target_test).sum().item()
            
            test_accuracy /= len(test_loader.dataset)
            test_accuracies.append(test_accuracy)
        return test_accuracies
    
    def train(self):
        if self.Non_iid==True:
            federated_trainset,testset = get_dataset(self.worker_num,self.alpha_size,self.alpha_label)
            train_iters = [torch.utils.data.DataLoader(federated_trainset[i], batch_size=self.batchsize, shuffle=True) for i in range(self.worker_num)]
            test_loader = DataLoader(testset,batch_size=self.test_batchsize,shuffle=False)
            train_infos = [Train_info(0.0, i) for i in range(self.worker_num)]
            time_thre = 5



        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5,0.5)
            ])

            trainset = torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
            testset = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
            

            time_thre = 5
            trains = torch.utils.data.random_split(trainset, [len(trainset) // self.worker_num for _ in range(self.worker_num)])
            # 分割した各データセットに対してDataLoaderオブジェクトを作る
            train_iters = [torch.utils.data.DataLoader(trains[i], batch_size=self.batchsize, shuffle=True) for i in range(self.worker_num)]
            test_loader = DataLoader(testset,batch_size=self.test_batchsize,shuffle=False)
            train_infos = [Train_info(0.0, i) for i in range(self.worker_num)]

        if self.cnn == "vgg":
            self.model = vgg13()

            if self.batchsize == 512:
                self.iter_time = 0.147
            if self.batchsize == 256:
                self.iter_time = 0.0781
            if self.batchsize == 128:
                self.iter_time = 0.0442
            if self.batchsize == 64:
                self.iter_time = 0.0337
        
        # モデルをcpuに送る，モデルをディープコピーする，コピーされたモデルを計算グラフから切り離す
        # self.model_zero = copy.deepcopy(self.model)
        # self.model_zero.zero()
        
       
        # クラアイントモデル
        models = []
        for _ in range(self.worker_num):
            model_copy = copy.deepcopy(self.model)
            models.append(model_copy)
        
        gradient_models = [None for i in range(self.worker_num)]

        time_pulls = (np.zeros((self.worker_num, self.wide_bandwidth)))
        
        communication_end = np.zeros(self.worker_num)
        train_nums = np.zeros(self.worker_num)
        train_lrs = np.full(self.worker_num, self.lr)

        wide_flags = np.zeros(self.worker_num,dtype=bool)
        wide_flags[:self.wide_node_size] = True
        estimated_times = np.full(self.worker_num, 60.0)
        gradient_times = np.zeros(self.worker_num)

        comm_time = self.model_size * 1.049 * 8 / 1000.0 + self.latency
        comm_time2 = (self.model_size * 1.049 * 8 / (1000.0 * self.wide_bandwidth) + self.latency) * self.wide_bandwidth
        comm_times = np.full((self.worker_num, self.worker_num), comm_time)
        comm_times[:self.wide_node_size, :] = np.full(self.worker_num, comm_time2 / self.wide_bandwidth)
        nums = np.arange(self.worker_num)

        output_time = 0
        output={}
        transition_accuracies = []

        for round_num in range(50000):
            if round_num%100==0:
                print(round_num)

            train_info = heapq.heappop(train_infos)
            start = train_info.time
            node_id = train_info.node_id

            if int(node_id)>=self.worker_num:
                node_id-=self.worker_num
                dest = (int(node_id) - (int(node_id) % self.worker_num)) // self.worker_num
                node_id %= self.worker_num
                gradient_models[node_id] = models[dest].copy(mode='copy').to_cpu()
                continue

            if math.floor(start)-output_time >= time_thre:
                output_time = math.floor(start)
                node_accurcies = self.test_models_max(models,test_loader,0,self.worker_num)
                mean_worker_accuracy = np.mean(node_accurcies)
                data = {
                    f"{output_time}":mean_worker_accuracy
                }
                transition_accuracies.append(data)

            while True:
                    dest = np.random.choice(nums)
                    if(dest != node_id):
                        break

           
            
            train_iter = train_iters[node_id]
            # learn modelの返り値は学習の実行時間
            if self.training_method=="gossip":
                time_diff = self.learn_model(models[node_id],train_iter,train_lrs[node_id])
            elif self.training_method=="mutual":
                time_diff = self.learnMutual_model(models,node_id,models[node_id],models[dest],train_iter,train_lrs[node_id])
            elif self.training_method=="collaborative":
                time_diff = self.learnEnsamble_model(models,node_id,models[node_id],models[dest],train_iter,train_lrs[node_id])



            cur = start+time_diff
            gradient_times[node_id] = cur

            train_iters[node_id] = train_iter
            train_nums[node_id] += 1
            
            # モデル平均化
            if int(train_nums[node_id])==int(self.T):
                train_nums[node_id] = 0
                
                if self.training_method=="gossip":
                    temp =copy.deepcopy(models[node_id])
                    with torch.no_grad():
                        for param1,param2 in zip(temp.parameters(),models[dest].parameters()):
                            param1.data = (param1.data+param2.data)/2
                    models[node_id] = temp

                if not wide_flags[dest]:
                    time_pulls[dest][0]=max(time_pulls[dest][0],cur)
                    time_pulls[dest][0]+=comm_time
                    cur = time_pulls[dest][0]
                else:
                    # 全ノードが広帯域であるためここにくる
                    time_pulls[dest] = np.clip(time_pulls[dest], cur, None)
                    # 自身は広帯域化どうか,絶対に広帯域である
                    if not wide_flags[node_id]:
                        cur += comm_time
                        time_pulls[dest][0] = cur
                        time_pulls[dest] = np.sort(time_pulls[dest])
                    else:
                        remaining_time = comm_time2  # 注意：comm_time2/self.wide_bandwidthではない！
                        tmp_pull_time = time_pulls[dest][0]
                        # print(tmp_pull_time)
                        for i in range(self.wide_bandwidth):
                            if i == 0:
                                continue
                            if time_pulls[dest][i] - tmp_pull_time > remaining_time / i:
                                tmp_pull_time += remaining_time / i
                                break
                            # print("before minus"+str(remaining_time))
                            remaining_time -= (time_pulls[dest][i] - tmp_pull_time) * i
                            # print(remaining_time)
                            tmp_pull_time = time_pulls[dest][i]
                            if i == self.wide_bandwidth - 1:
                                tmp_pull_time += remaining_time / self.wide_bandwidth
                        cur = tmp_pull_time
                        np.clip(time_pulls[dest], tmp_pull_time, None)
            
           
            heapq.heappush(train_infos, Train_info(cur, node_id))

        with open(f"{self.training_method}_wide_{self.wide_node_size}_non_iid{self.Non_iid}.json","w") as f:
                    json.dump(transition_accuracies,f)
                    f.write("\n")
        

if __name__ == "__main__":
    args = parse_argument()
    Train(args).train()













        

        















