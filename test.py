import tqdm
from lib.config import cfg,args
from torchinfo import summary
import cv2
from lib.networks import make_network
from lib.networks.cVAE import Network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
from lib.train.trainers.network_wrapper import NetworkWrapper
import torch.multiprocessing
import torch
import torch.distributed as dist
import os

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(cfg):
    network = Network() 
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    #load model
    model_dir = os.path.join('data/trained_model/StructureNeRF/2048rays/latest.pth')
    trained_model = torch.load(model_dir)
    network.load_state_dict(trained_model['net'])
    epoch = trained_model['epoch'] + 1
    trainer.val(epoch, val_loader, network, evaluator)
    return network


def main():
    test(cfg)


if __name__ == "__main__":
    main()
