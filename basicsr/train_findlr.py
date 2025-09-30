# source evssm-venv/bin/activate
# python basicsr/train.py -opt options/train/GoPro.yml & echo $!
# python basicsr/train.py -opt options/train/Realblur.yml & echo $!
import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse
from tqdm import tqdm  # 添加这一行
# 在 import 部分后添加
import matplotlib.pyplot as plt
import numpy as np

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=False, help='Path to option YAML file.',default='options/train/GoPro.yml')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # tb_logger = init_tb_logger(log_dir=f'./logs/{opt['name']}') #mkdir logs @CLY
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    num_iter_per_epoch = 0  # 添加初始化
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    # return train_loader, train_sampler, val_loader, total_epochs, total_iters
    return train_loader, train_sampler, val_loader, total_epochs, total_iters, num_iter_per_epoch  # 修改返回值

# 在 main() 函数前添加这个新函数
def find_learning_rate(opt, model, train_loader, logger):
    """Find optimal learning rate using LR range test"""
    logger.info('Starting learning rate finder...')
    
    # LR finder parameters
    min_lr = 1e-7
    max_lr = 10
    num_iter = min(500, len(train_loader))  # Use 500 iterations or one epoch
    
    # Setup
    model.net_g.train()
    lrs = []
    losses = []
    smoothed_losses = []
    beta = 0.98  # Smoothing factor
    
    # Calculate LR multiplication factor
    lr_mult = (max_lr / min_lr) ** (1.0 / num_iter)
    lr = min_lr
    
    # Create prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}')
    
    prefetcher.reset()
    train_data = prefetcher.next()
    
    # Progress bar
    pbar = tqdm(total=num_iter, desc='LR Finder', ncols=100)
    
    best_loss = float('inf')
    
    for iteration in range(num_iter):
        if train_data is None:
            break
            
        # Set learning rate
        for param_group in model.optimizer_g.param_groups:
            param_group['lr'] = lr
        
        # Forward pass using model's feed_data and get loss
        model.feed_data(train_data, is_val=False)
        model.optimizer_g.zero_grad()
        
        # 使用torch.no_grad()和torch.cuda.amp来减少显存使用
        with torch.cuda.amp.autocast(enabled=False):  # 禁用自动混合精度以保持稳定
            # 直接使用模型的前向传播
            model.output = model.net_g(model.lq)
            
            # Calculate losses using model's loss functions
            l_total = 0
            loss_dict = {}
            
            # Pixel loss
            if hasattr(model, 'cri_pix') and model.cri_pix is not None:
                l_pix = model.cri_pix(model.output, model.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix.item()
            
            # FFT loss
            if hasattr(model, 'cri_fft') and model.cri_fft is not None:
                l_fft = model.cri_fft(model.output, model.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft.item()
                
            loss = l_total.item()
        
        # Skip first iteration for best_loss initialization
        if iteration == 0:
            best_loss = loss
        else:
            if loss < best_loss:
                best_loss = loss
            
            # Stop if loss explodes
            if loss > 10 * best_loss or np.isnan(loss) or np.isinf(loss):
                logger.info(f'Stopping early due to exploding loss at LR={lr:.2e}')
                break
        
        # Record
        lrs.append(lr)
        losses.append(loss)
        
        # Calculate smoothed loss
        if iteration == 0:
            smoothed_loss = loss
        else:
            smoothed_loss = beta * smoothed_losses[-1] + (1 - beta) * loss
        smoothed_losses.append(smoothed_loss)
        
        # Backward pass
        l_total.backward()
        
        # Gradient clipping (optional, helps stability)
        torch.nn.utils.clip_grad_norm_(model.net_g.parameters(), max_norm=1.0)
        
        model.optimizer_g.step()
        
        # 清理中间变量以释放显存
        del model.output
        if 'l_pix' in locals():
            del l_pix
        if 'l_fft' in locals():
            del l_fft
        del l_total
        
        # 定期清理显存
        if iteration % 10 == 0:
            torch.cuda.empty_cache()
        
        # Update learning rate
        lr *= lr_mult
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{loss:.4f}'})
        
        # Get next batch
        train_data = prefetcher.next()
          
    pbar.close()
    
    # Only plot if we have enough data points
    if len(lrs) > 20:
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Raw losses
        ax1.plot(lrs, losses, alpha=0.3, label='Raw Loss')
        ax1.plot(lrs, smoothed_losses, label='Smoothed Loss')
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Rate Finder - Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Loss gradient
        gradients = []
        for i in range(1, len(smoothed_losses)):
            gradient = (smoothed_losses[i] - smoothed_losses[i-1]) / (lrs[i] - lrs[i-1])
            gradients.append(gradient)
        
        if gradients:
            ax2.plot(lrs[1:], gradients)
            ax2.set_xscale('log')
            ax2.set_xlabel('Learning Rate')
            ax2.set_ylabel('Loss Gradient')
            ax2.set_title('Learning Rate Finder - Gradient')
            ax2.grid(True, alpha=0.3)
        
        # Save plot
        save_dir = osp.join(opt['path']['experiments_root'], opt['name'])
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, 'lr_finder.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f'LR finder plot saved to {save_path}')
        
        # Find suggested learning rate
        # Method 1: Steepest descent (minimum gradient)
        min_gradient_idx = None
        min_gradient = float('inf')
        
        # Look in the middle 80% of the range to avoid edges
        start_idx = len(smoothed_losses) // 10
        end_idx = len(smoothed_losses) * 9 // 10
        
        for i in range(start_idx, min(end_idx, len(smoothed_losses) - 5)):
            # Calculate gradient over a window
            gradient = (smoothed_losses[i+5] - smoothed_losses[i]) / 5
            if gradient < min_gradient:
                min_gradient = gradient
                min_gradient_idx = i
        
        # Method 2: Find where loss starts increasing
        min_loss_idx = np.argmin(smoothed_losses)
        
        if min_gradient_idx:
            suggested_lr_steep = lrs[min_gradient_idx]
            suggested_lr_min = lrs[min_loss_idx] if min_loss_idx > 0 else suggested_lr_steep
            
            logger.info('='*60)
            logger.info('Learning Rate Finder Results:')
            logger.info(f'  Steepest descent LR: {suggested_lr_steep:.2e}')
            logger.info(f'  Minimum loss LR: {suggested_lr_min:.2e}')
            logger.info(f'  Conservative LR (recommended): {suggested_lr_steep/10:.2e}')
            logger.info(f'  Aggressive LR: {suggested_lr_steep:.2e}')
            logger.info('='*60)
            
            # Save results to file
            results_path = osp.join(save_dir, 'lr_finder_results.txt')
            with open(results_path, 'w') as f:
                f.write('Learning Rate Finder Results\n')
                f.write('='*40 + '\n')
                f.write(f'Steepest descent LR: {suggested_lr_steep:.2e}\n')
                f.write(f'Minimum loss LR: {suggested_lr_min:.2e}\n')
                f.write(f'Conservative LR (recommended): {suggested_lr_steep/10:.2e}\n')
                f.write(f'Aggressive LR: {suggested_lr_steep:.2e}\n')
                f.write('\nRecommendation: Use the conservative LR for stable training.\n')
            logger.info(f'Results saved to {results_path}')
    else:
        logger.warning('Not enough data points collected for LR finding. Try increasing num_iter.')
    
    return lrs, losses, smoothed_losses

# def main():
#     # parse options, set distributed setting, set ramdom seed
#     opt = parse_options(is_train=True)

#     torch.backends.cudnn.benchmark = True
#     # torch.backends.cudnn.deterministic = True

#     # automatic resume ..
#     state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
#     import os
#     try:
#         states = os.listdir(state_folder_path)
#     except:
#         states = []

#     resume_state = None
#     if len(states) > 0:
#         print('!!!!!! resume state .. ', states, state_folder_path)
#         max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
#         resume_state = os.path.join(state_folder_path, max_state_file)
#         opt['path']['resume_state'] = resume_state

#     # load resume states if necessary
#     if opt['path'].get('resume_state'):
#         device_id = torch.cuda.current_device()
#         resume_state = torch.load(
#             opt['path']['resume_state'],
#             map_location=lambda storage, loc: storage.cuda(device_id))
#     else:
#         resume_state = None

#     # mkdir for experiments and logger
#     if resume_state is None:
#         make_exp_dirs(opt)
#         if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
#                 'name'] and opt['rank'] == 0:
#             mkdir_and_rename(osp.join('tb_logger', opt['name']))

#     # initialize loggers
#     logger, tb_logger = init_loggers(opt)

#     # create train and validation dataloaders
#     result = create_train_val_dataloader(opt, logger)
#     # train_loader, train_sampler, val_loader, total_epochs, total_iters = result
#     train_loader, train_sampler, val_loader, total_epochs, total_iters, num_iter_per_epoch = result  # 修改接收

#     # create model
#     if resume_state:  # resume training
#         check_resume(opt, resume_state['iter'])
#         model = create_model(opt)
#         model.resume_training(resume_state)  # handle optimizers and schedulers
#         logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
#                     f"iter: {resume_state['iter']}.")
#         start_epoch = resume_state['epoch']
#         current_iter = resume_state['iter']
#     else:
#         model = create_model(opt)
#         start_epoch = 0
#         current_iter = 0

#         # 学习率查找
#         if opt.get('find_lr', False):
#             find_learning_rate(opt, model, train_loader, logger)
#             logger.info('Learning rate finding complete. Check the plot and update your config.')
#             return  # Exit after finding LR
def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 添加学习率查找模式的检查
    if opt.get('find_lr', False):
        # 跳过自动恢复
        resume_state = None
        make_exp_dirs(opt)
    else:
        # 原始的自动恢复代码
        state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
        import os
        try:
            states = os.listdir(state_folder_path)
        except:
            states = []

        resume_state = None
        if len(states) > 0:
            print('!!!!!! resume state .. ', states, state_folder_path)
            max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
            resume_state = os.path.join(state_folder_path, max_state_file)
            opt['path']['resume_state'] = resume_state

        # load resume states if necessary
        if opt['path'].get('resume_state'):
            device_id = torch.cuda.current_device()
            resume_state = torch.load(
                opt['path']['resume_state'],
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            resume_state = None

        # mkdir for experiments and logger
        if resume_state is None:
            make_exp_dirs(opt)
            if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                    'name'] and opt['rank'] == 0:
                mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters, num_iter_per_epoch = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
        
        # Add learning rate finder here
        if opt.get('find_lr', False):
            find_learning_rate(opt, model, train_loader, logger)
            logger.info('Learning rate finding complete. Check the plot and update your config.')
            return  # Exit after finding LR   
    
    

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    # 添加总进度条
    pbar = tqdm(total=total_iters, initial=current_iter, desc='Training Progress', ncols=120, unit='iter', position=0)
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        # 添加epoch进度条
        epoch_pbar = tqdm(total=num_iter_per_epoch, desc=f'Epoch {epoch}/{total_epochs}', 
                         ncols=100, unit='batch', position=1, leave=False)
        epoch_iter = 0

        
        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data, is_val=False)
            result_code = model.optimize_parameters(current_iter, tb_logger)
            # if result_code == -1 and tb_logger:
            #     print('loss explode .. ')
            #     exit(0)
            iter_time = time.time() - iter_time
            
            # 更新进度条（添加以下代码）
            pbar.update(1)
            epoch_pbar.update(1)
            epoch_iter += 1
            
            # 获取当前loss并更新进度条描述
            current_log = model.get_current_log()
            loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in current_log.items() if 'loss' in k.lower()])
            lr = model.get_current_learning_rate()[0] if isinstance(model.get_current_learning_rate(), list) else model.get_current_learning_rate()
            
            # 计算ETA
            elapsed = time.time() - start_time
            initial_iter = resume_state['iter'] if resume_state else 0
            iterations_done = current_iter - initial_iter
            if iterations_done > 0:
                eta_seconds = (elapsed / iterations_done) * (total_iters - current_iter)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "N/A"
            # 添加这两行来更新进度条显示
            pbar.set_description(f'Training [ETA: {eta_str}] [LR: {lr:.2e}]')
            epoch_pbar.set_postfix_str(loss_str)
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                # print('msg logger .. ', current_iter)
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0 or current_iter == 1000):
            # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                tqdm.write(f'\n{"="*50}\nValidation at iter {current_iter}\n{"="*50}')  # 添加这一行
                
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image )
                # 添加验证结果显示
                val_metrics = model.get_current_log()
                metric_str = ', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items() if 'psnr' in k.lower() or 'ssim' in k.lower()])
                if metric_str:
                    tqdm.write(f'Validation Results: {metric_str}\n{"="*50}\n')
                
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)


            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch_pbar.close()  # 添加这一行
        epoch += 1

    # end of epoch
    pbar.close()  # 添加这一行
    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        metric = model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'], rgb2bgr, use_image)
        # if tb_logger:
        #     print('xxresult! ', opt['name'], ' ', metric)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY']='epoll1'
    main()
