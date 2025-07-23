import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.version
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from torch_scatter import scatter_add, scatter_mean

import arguments
import utils
from dataset import PDBbindDataset
from dataset_2 import PDBbindDataset_2
from model import DeepRL_3D_model

import pandas as pd
from pathlib import Path

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


print(torch.__version__)
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

# sys.exit()


def sample_to_device(sample, device):
    # device = f"cuda:{device}"
    sample["whole"].to(device)
    sample["traj"].to(device)
    return

'''         
model=model,
args=args,
optimizer=optimizer,
data=train_data,
train=True,
device=gpu,
scaler=scaler,
'''


def train(model, args, optimizer, data, data_2, train, device=None, scaler=None):
    model.train() if train else model.eval()

    total_losses, vae_losses, type_losses, dist_losses, ssl_losses = [], [], [], [], []
    while True:
        # t=t+1
        # print("======True=====",t)
        # if t==305:
        #     exit()
        sample = next(data, None) 
        sample_2 = next(data_2, None)
        if sample is None:
            break
        if sample_2 is None:
            break

        sample_to_device(sample, device)
        sample_to_device(sample_2, device)
        
        #print("======sample====",sample.get("traj").dist_ll_output.sum())
        #exit()



        if args.autocast:
            with amp.autocast():
                try:
                    total_loss_1, vae_loss_1, type_loss_1, dist_loss_1, ssl_loss_1 = model(sample, device)
                    total_loss_2, vae_loss_2, type_loss_2, dist_loss_2, ssl_loss_2 = model(sample_2, device)
                    total_loss = total_loss_1 + total_loss_2

                    vae_loss = vae_loss_1 + vae_loss_2
                    type_loss = type_loss_1 + type_loss_2
                    dist_loss = dist_loss_1 + dist_loss_2
                    # loss_label_reg = loss_label_reg_1 + loss_label_reg_2
                    if ssl_loss_1 is None:
                        ssl_loss = None
                    else:
                        ssl_loss = ssl_loss_1 + ssl_loss_2
                except Exception as e:
                    print(traceback.format_exc())
                    exit()
        else:
            try:
                #print("=======sample=====",sample)
                # print("=======sample_whole=====",sample["whole"]["ligand"]["x"])
                total_loss_1, vae_loss_1, type_loss_1, dist_loss_1, ssl_loss_1 = model(sample, device)
                total_loss_2, vae_loss_2, type_loss_2, dist_loss_2, ssl_loss_2 = model(sample_2, device)
                total_loss = total_loss_1 + total_loss_2

                vae_loss = vae_loss_1 + vae_loss_2
                type_loss = type_loss_1 + type_loss_2
                dist_loss = dist_loss_1 + dist_loss_2
                # loss_label_reg = loss_label_reg_1 + loss_label_reg_2
                if ssl_loss_1 is None:
                    ssl_loss = None
                else:
                    ssl_loss = ssl_loss_1 + ssl_loss_2
                #print("======vae_loss",total_loss)
                #sys.exit()
            except Exception as e:
                print(traceback.format_exc())
                exit()
        if train:
            optimizer.zero_grad()
            if args.autocast:
                scaler.scale(total_loss).backward()
                # scaler.scale(total_loss_1).backward(retain_graph=True)
                # scaler.scale(total_loss_2).backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                # total_loss_1.backward(retain_graph=True)
                # total_loss_2.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()

        vae_losses.append(vae_loss.data.cpu().numpy())
        type_losses.append(type_loss.data.cpu().numpy())
        dist_losses.append(dist_loss.data.cpu().numpy())
        total_losses.append(total_loss.data.cpu().numpy())
        if args.ssl:
            ssl_losses.append(ssl_loss.data.cpu().numpy())

    vae_losses = np.mean(np.array(vae_losses))
    type_losses = np.mean(np.array(type_losses))
    dist_losses = np.mean(np.array(dist_losses))
    total_losses = np.mean(np.array(total_losses))
    if args.ssl:
        ssl_losses = np.mean(np.array(ssl_losses))
        return total_losses, vae_losses, type_losses, dist_losses, ssl_losses

    return total_losses, vae_losses, type_losses, dist_losses, 0.0


def main_worker(gpu, ngpus_per_node, args):
    rank = gpu
    print("Rank:", rank, flush=True)

    # Path
    save_dir = utils.get_abs_path(args.save_dir)
    #data_dir = utils.get_abs_path(args.data_dir)
    if args.restart_file:
        restart_file = utils.get_abs_path(args.restart_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Dataloader
    train_dataset = PDBbindDataset(args, mode="train")
    valid_dataset = PDBbindDataset(args, mode="valid")

    train_dataset_2 = PDBbindDataset_2(args, mode="train")  
    valid_dataset_2 = PDBbindDataset_2(args, mode="valid")   

    #dataloder label
    # df = pd.read_csv(
    # index_dir,
    # sep=r"\s+",
    # comment="#",
    # header=None,
    # )
    # name_label_map = {}
    # for _, row in df.iterrows():
    #     name_label_map[row[0]] = row[3]


    # ###
    # data = name_label_map
    # a, b = 0, 1
    # min_value = min(name_label_map.values())
    # max_value = max(name_label_map.values())

    # #  [a, b]
    # normalized_data = {k: round(a + ((v - min_value) * (b - a)) / (max_value - min_value), 4) for k, v in data.items()}

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=8, pin_memory=True #, sampler=train_sampler
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, num_workers=8, pin_memory=True, shuffle=False #,sampler=valid_sampler,
    )

    train_dataloader_2 = DataLoader(train_dataset_2, batch_size=1, num_workers=8, pin_memory=True #, sampler=train_sampler
    )
    valid_dataloader_2 = DataLoader(valid_dataset_2, batch_size=1, num_workers=8, pin_memory=True, shuffle=False #,sampler=valid_sampler,
    )

    N_TRAIN_DATA = len(train_dataset)
    N_VALID_DATA = len(valid_dataset)
    N_TRAIN_DATA_2 = len(train_dataset_2)
    N_VALID_DATA_2 = len(valid_dataset_2)
    #if not args.restart_file and rank == 0:
    print("Train dataset length: ", N_TRAIN_DATA, flush=True)
    print("Validation dataset length: ", N_VALID_DATA, flush=True)
    print("Train dataset_2 length: ", N_TRAIN_DATA_2, flush=True)
    print("Validation dataset_2 length: ", N_VALID_DATA_2, flush=True)
    print(utils.text_filling("Finished Loading Datasets"), flush=True)

    # Model initialize
    model = DeepRL_3D_model(args)
    model = model.to("cuda:0")

    print(
        "Number of Parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),  #numel() 
        flush=True,
    )
    print(utils.text_filling("Finished Loading Model"), flush=True)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Scaler (AMP)
    if args.autocast:
        scaler = amp.GradScaler() ‌
    else:
        scaler = None

    # VAE loss annealing
    vae_coeff_init, vae_coeff_final = args.vae_loss_coeff   #vae_loss_coeff=[0.0, 1.0]

    # Train
    # if args.restart_file:
    #     start_epoch = int(args.restart_file.split("_")[-1].split(".")[0])
    # else:
    #     start_epoch = 0
    start_epoch = 0

    best_epoch = 0
    min_valid_loss = 1e6
    lr_tick = 0
    for epoch in range(start_epoch, args.num_epochs):
        # if epoch == 1:
        #     print("======epoch=0====")
        #     exit()
        if epoch == 0: # and rank == 0:
            print(
                "EPOCH || "
                + "TRA_VAE | "
                + "TRA_SSL | "
                + "TRA_TYPE | "
                + "TRA_DIST | "
                + "TRA_TOT || "
                #+ "TRA_AFFI || "
                + "VAL_VAE | "
                + "VAL_SSL | "
                + "VAL_TYPE | "
                + "VAL_DIST | "
                + "VAL_TOT || "
                #+ "VAL_AFFI || "
                + "TIME/EPOCH | "
                + "LR | "
                + "BEST_EPOCH",
                flush=True,
            )

        train_data = iter(train_dataloader)
        valid_data = iter(valid_dataloader)

        train_data_2 = iter(train_dataloader_2)
        valid_data_2 = iter(valid_dataloader_2)

        # for x in train_data:
        #     print("=====================",x)
        #     exit()

        # KL annealing
        args.vae_coeff = vae_coeff_final + (vae_coeff_init - vae_coeff_final) * (
            (1 - args.vae_loss_beta) ** (epoch+1)
        )

        # print("======args.vae_coeff=======",args.vae_coeff)
        # print("======args.vae_coeff=======",epoch)
        # sys.exit()

        st = time.time()

        (
            train_total_losses,
            train_vae_losses,
            train_type_losses,
            train_dist_losses,
            train_ssl_losses,
            #train_affinity_pre
        ) = train(
            model=model,
            args=args,
            optimizer=optimizer,
            data=train_data,
            data_2=train_data_2,
            train=True,
            device=gpu,
            scaler=scaler,
            # name_label_map=normalized_data#name_label_map
        )

        # validation process
        (
            valid_total_losses,
            valid_vae_losses,
            valid_type_losses,
            valid_dist_losses,
            valid_ssl_losses,
            #valide_affinity_pre
        ) = train(
            model=model,
            args=args,
            optimizer=optimizer,
            data=valid_data,
            data_2=valid_data_2,
            train=False,
            device=gpu,
            scaler=scaler,
            # name_label_map=normalized_data#name_label_map
        )

        et = time.time()

        if valid_total_losses < min_valid_loss:
            min_valid_loss = valid_total_losses
            best_epoch = epoch
            lr_tick = 0
        else:
            lr_tick += 1

        if lr_tick >= args.lr_tolerance:
            for param_group in optimizer.param_groups:
                lr = param_group["lr"]
                if lr > args.lr_min:
                    param_group["lr"] = lr * args.lr_decay

        ### TODO  #
        if lr_tick > 40:
            print("No longer model is learning: training stop")
            exit()

        #if rank == 0:
        print(
            f"{epoch} || "
            + f"{train_vae_losses:.3f} | "
            + f"{train_ssl_losses:.3f} | "
            + f"{train_type_losses:.3f} | "
            + f"{train_dist_losses:.3f} | "
            + f"{train_total_losses:.3f} || "
            #+ f"{train_affinity_pre:.3f} || "
            + f"{valid_vae_losses:.3f} | "
            + f"{valid_ssl_losses:.3f} | "
            + f"{valid_type_losses:.3f} | "
            + f"{valid_dist_losses:.3f} | "
            + f"{valid_total_losses:.3f} || "
            #+ f"{valide_affinity_pre:.3f} || "
            + f"{(et - st):.2f} | "
            + f"{[group['lr'] for group in optimizer.param_groups][0]:.4f} | "
            + f"{best_epoch}{'*' if lr_tick==0 else ''}",
            flush=True,
        )

        # Save model
        name = os.path.join(save_dir, f"save_{epoch}.pt")
        save_every = 1 if not args.save_every else args.save_every
        if epoch % save_every == 0: # and rank == 0:
            torch.save(model.state_dict(), name)


def main():
    now = datetime.now()
    print(
        f"Train starts at {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"
    )

    args = arguments.train_args_parser()
    d = vars(args)
    print(utils.text_filling("PARAMETER SETTINGS"), flush=True)
    for a in d:
        print(a, "=", d[a])
    print(80 * "#", flush=True)

    args.master_port = utils.find_free_port()


    args.distributed = args.world_size > 1

    #os.environ["CUDA_VISIBLE_DEVICES"] = utils.get_cuda_visible_devices(args.world_size) 

    if args.distributed:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(
                args.world_size,
                args,
            ),
        )
    else:
        main_worker(0, args.world_size, args)  #GPU


if __name__ == "__main__":
    main()
