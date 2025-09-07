import os

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from pathlib import Path
from options import args_parser
from utils import normalize_augment,setup_logger,save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from dataset import VideoSequenceDataset,TestDataset
from model import SwinIR, CharbonnierLoss,validate_model
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = args_parser()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir)
    logger.info(f'Training started with device: {device}')
    logger.info(f'Arguments: {args}')
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard_logs'))



    train_dataset = VideoSequenceDataset(file_root=args.train_dataset,sequence_length=1,
                                         crop_size=128, epoch_size= args.max_number_patches,
                                         random_shuffle= True,temp_stride= -1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    # testset

    test_dataset = TestDataset(data_dir= args.val_data)
    test_loader = DataLoader(test_dataset ,batch_size=1)
    # model
    model = SwinIR(upscale= 1,in_chans=3,
                       img_size=128,
                       window_size=8,
                       img_range=1.0,
                       depths=[6,6,6,6,6,6],
                       embed_dim=180,
                       num_heads=[6,6,6,6,6,6],
                       mlp_ratio=2,
                       upsampler= None,
                       resi_connection="1conv")
    model = model.to(device)
    criterion = CharbonnierLoss(1e-9).to(device)
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))
    optimizer = Adam(optim_params, lr=2e-4,
                                betas= (0.9,0.999),
                                weight_decay=0)
    schedular = lr_scheduler.MultiStepLR(optimizer,[800000, 1200000, 1400000, 1500000, 1600000],0.5)
    current_step = 0
    logger.info('Starting training...')
    for epoch in range(50):
        logger.info(f'Training Round: {epoch + 1}')
        print(f'\n | Training Round : {epoch + 1} |\n')
        for i,(seq,gt) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            img_train, gt_train = normalize_augment(seq,0)
            device = next(model.parameters()).device
            img_train = img_train.to(device, non_blocking=True)
            gt_train = gt_train.to(device, non_blocking=True)
            N, _, H, W = img_train.size()
            stdn = torch.empty((N, 1, 1, 1), device=device).uniform_(
                args.noise_level[0], args.noise_level[1]
            )
            noise = torch.zeros_like(img_train)
            noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
            imgn_train = img_train + noise
            #noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True)
            out_train = model(imgn_train)
            loss = 1.0 * criterion(out_train,gt_train)
            loss.backward()
            optimizer.step()
            current_step += 1
            schedular.step(current_step)
        if epoch % args.val_freq == 0:
            logger.info('Running validation...')
            model.eval()
            psnr =validate_model(
                    args = args,
                    model = model,
                    dataset_val = test_loader,
                    valnoisestd = args.test_noise,
                    logger = logger,
                    writer = writer,
                    epoch = epoch,
                    lr = optimizer.param_groups[0]['lr'],
                    trainimg=img_train,
                    device = device
            )

            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch: {epoch + 1:3d}, Step: {current_step:8d}, '
                            f'LR: {lr:.3e}, Loss: {loss.item():.6f}, PSNR: {psnr:.4f}')

                # TensorBoard logging
            writer.add_scalar('Train/Loss', loss.item(), current_step)
            writer.add_scalar('Train/PSNR', psnr, current_step)
            writer.add_scalar('Train/Learning_Rate', lr, current_step)

        if epoch % args.save_freq == 0:
            logger.info(f'Saving checkpoint at step {current_step}')
            checkpoint_path = save_checkpoint(model, optimizer, schedular, epoch,
                                                  current_step, args.save_dir)
            logger.info(f'Checkpoint saved to: {checkpoint_path}')
    writer.close()
    logger.info('Training completed!')

if __name__ == '__main__':
    main()





