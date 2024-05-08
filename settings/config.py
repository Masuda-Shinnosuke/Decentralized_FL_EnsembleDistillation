import argparse

def parse_argument():

    parser = argparse.ArgumentParser(description="Learning gossip")
    parser.add_argument("-t","--training_method",default="collaborative",help="The training method to use: gossip,collaborative")
    parser.add_argument("-d","--dataset",default="cifar10",help='The dataset to use: cifar10, cifar100, fashion mnist, svhn')
    parser.add_argument("-c","--cnn",default="vgg",help="The CNN to use: vgg,resnet")
    parser.add_argument("-B","--batchsize", type=int, default=128, help='Learning minibatch size')
    parser.add_argument("-g","--gpu", type=int, default=-1, help='GPU ID (negative value indicates CPU')
    parser.add_argument("-b","--test_batchsize",type=int,default=100,help="Validation minibatch size")
    parser.add_argument("-n","--non_iid",type=int,default=2,help="Wheather Non-iid or not")
    args = parser.parse_args()

    return args