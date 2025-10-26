from utils.data import *
from utils.metric import *
from argparse import ArgumentParser
import torch
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import *
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time

# TPU相关导入
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True
    print("✓ TPU库加载成功")
except ImportError:
    TPU_AVAILABLE = False
    print("✗ TPU库未安装，将使用GPU/CPU")

os.environ['CUDA_VISIBLE_DEVICES']="0"

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default='/dataset/IRSTD-1k')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight-path', type=str, default='/MSHNet/weight/IRSTD-1k_weight.tar')
    
    # TPU相关参数
    parser.add_argument('--use-tpu', action='store_true', help='使用TPU训练')
    parser.add_argument('--num-cores', type=int, default=8, help='TPU核心数量')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args, device=None):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode
        self.use_tpu = args.use_tpu and TPU_AVAILABLE

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        # 设置设备
        if device is not None:
            # TPU多核训练时，device由XLA传入
            self.device = device
            print(f"Using TPU device: {device}")
        elif self.use_tpu and TPU_AVAILABLE:
            # 单核TPU
            self.device = xm.xla_device()
            print(f"Using TPU device: {self.device}")
        elif torch.cuda.is_available():
            # GPU
            self.device = torch.device('cuda')
            print("Using GPU for training")
        else:
            # CPU
            self.device = torch.device('cpu')
            print("CUDA is not available, using CPU instead")

        # 数据加载器
        if self.use_tpu:
            # TPU使用特殊的数据加载器
            self.train_loader = Data.DataLoader(
                trainset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                drop_last=True,
                num_workers=4  # TPU推荐使用多进程
            )
            self.val_loader = Data.DataLoader(
                valset, 
                batch_size=1, 
                drop_last=False,
                num_workers=4
            )
        else:
            self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True)
            self.val_loader = Data.DataLoader(valset, 1, drop_last=False)

        model = MSHNet(3)

        if args.multi_gpus and not self.use_tpu:
            if torch.cuda.device_count() > 1:
                print('use '+str(torch.cuda.device_count())+' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        
        model.to(self.device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = ''
                checkpoint = torch.load(check_folder+'/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']+1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                # 使用相对路径，适配不同环境（本地/Colab）
                weight_base_dir = './weight'
                os.makedirs(weight_base_dir, exist_ok=True)
                self.save_folder = osp.join(weight_base_dir, 'MSHNet-%s'%(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))))
                os.makedirs(self.save_folder, exist_ok=True)
        if args.mode=='test':
          
            weight = torch.load(args.weight_path)
            self.model.load_state_dict(weight['state_dict'])
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1
        

    def train(self, epoch):
        self.model.train()
        
        # TPU使用ParallelLoader包装数据加载器
        if self.use_tpu:
            train_loader = pl.ParallelLoader(self.train_loader, [self.device]).per_device_loader(self.device)
        else:
            train_loader = self.train_loader
            
        tbar = tqdm(train_loader)
        losses = AverageMeter()
        tag = False
        
        for i, (data, mask) in enumerate(tbar):
  
            data = data.to(self.device)
            labels = mask.to(self.device)

            if epoch>self.warm_epoch:
                tag = True

            masks, pred = self.model(data, tag)
            loss = 0

            loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
            for j in range(len(masks)):
                if j>0:
                    labels = self.down(labels)
                loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)
                
            loss = loss / (len(masks)+1)
        
            self.optimizer.zero_grad()
            loss.backward()
            
            # TPU需要特殊的优化器步骤
            if self.use_tpu:
                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        
        # TPU使用ParallelLoader
        if self.use_tpu:
            val_loader = pl.ParallelLoader(self.val_loader, [self.device]).per_device_loader(self.device)
        else:
            val_loader = self.val_loader
            
        tbar = tqdm(val_loader)
        tag = False
        
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
    
                data = data.to(self.device)
                mask = mask.to(self.device)

                if epoch>self.warm_epoch:
                    tag = True

                loss = 0
                _, pred = self.model(data, tag)
                # loss += self.loss_fun(pred, mask,self.warm_epoch, epoch)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            
            FA, PD = self.PD_FA.get(len(val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            
            if self.mode == 'train':
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    # TPU保存模型需要先移到CPU
                    if self.use_tpu:
                        xm.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    else:
                        torch.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, PD[0], FA[0] * 1000000))
                        
                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
                
                if self.use_tpu:
                    xm.save(all_states, self.save_folder+'/checkpoint.pkl')
                else:
                    torch.save(all_states, self.save_folder+'/checkpoint.pkl')
                    
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')


def _mp_fn(index, args):
    """TPU多核训练的入口函数"""
    device = xm.xla_device()
    trainer = Trainer(args, device=device)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            # 只在主进程上运行验证
            if xm.is_master_ordinal():
                trainer.test(epoch)
    else:
        trainer.test(1)

         
if __name__ == '__main__':
    args = parse_args()

    if args.use_tpu and TPU_AVAILABLE:
        # TPU多核训练
        print(f"启动TPU训练，使用{args.num_cores}个核心")
        xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores, start_method='fork')
    else:
        # GPU/CPU训练
        trainer = Trainer(args)
        
        if trainer.mode=='train':
            for epoch in range(trainer.start_epoch, args.epochs):
                trainer.train(epoch)
                trainer.test(epoch)
        else:
            trainer.test(1)

