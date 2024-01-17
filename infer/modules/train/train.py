import csv
import math
import os
import shutil
import sys
import logging

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import datetime

from infer.lib.train import utils

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.cuda.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee

global_step = 0
bestEpochStep=0
lastValue=1
lowestValue = {"step": 0, "value": float("inf"), "epoch": 0}
dirtyTb = []
dirtyValues = []
dirtySteps = []
dirtyEpochs = []
continued = False

class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    logger.info("\n")
    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(
    rank,
    n_gpus,
    hps,
):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        pass
    elif torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global bestEpochStep, lastValue, lowestValue, continued
        if hps.if_retrain_collapse:
            if os.path.exists(f"{hps.model_dir}/col"):
                with open(f"{hps.model_dir}/col", 'r') as f:
                    bestEpochStep=global_step = int(f.readline().split(',')[0])
                os.remove(f"{hps.model_dir}/col")
                continued = True
            if not os.path.exists(f"{hps.model_dir}/col") and os.path.exists(f"{hps.model_dir}/fitness.csv"):
                latest = ""
                with open(f'{hps.model_dir}/fitness.csv', 'r') as f:
                    for line in f:
                        if line.strip() != "":
                            latest = line.split(',')
                global_step = int(latest[1])
                lastValue = float(latest[2])
                lowestValue = {"step": int(latest[1]), "value": float(latest[2]), "epoch": int(latest[0])}
                continued = True
        else:
            global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )  ##测试不加载优化器
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info if hps.if_f0 == 1 else info[:2] + info[4:]
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0 and not hps.if_stop_on_fit:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f}, loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                tensorboard_summarize(writer, mel, y_mel, y_hat_mel, loss_disc, losses_disc_r, losses_disc_g, grad_norm_d, loss_mel, loss_kl, loss_fm, losses_gen, loss_gen_all, grad_norm_g, lr)
        global_step += 1
    # /Run steps

    if hps.save_every_epoch != 0 and epoch % hps.save_every_epoch == 0 and rank == 0:
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "G_{}.pth".format(global_step if hps.if_latest == 0 else 2333333)),
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "D_{}.pth".format(global_step if hps.if_latest == 0 else 2333333)),
        )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    global dirtyTb, dirtySteps, dirtyValues, dirtyEpochs, bestEpochStep, lastValue, continued

    if rank == 0 and hps.if_retrain_collapse and loss_gen_all / lastValue < 0.25:
        logger.warning("Mode collapse detected, model quality may be hindered. More information here: https://rentry.org/RVC_making-models#mode-collapse")
        logger.warning(f'loss_gen_all={loss_gen_all.item()}, last value={lastValue}, drop % {loss_gen_all.item() / lastValue * 100}')
        tensorboard_summarize(writer_eval, mel, y_mel, y_hat_mel, loss_disc, losses_disc_r, losses_disc_g, grad_norm_d, loss_mel, loss_kl, loss_fm, losses_gen, loss_gen_all, grad_norm_g, optim_g.param_groups[0]["lr"])
        if hps.if_retrain_collapse:
            logger.info("Restarting training from last fit epoch..." if hps.train.batch_size else "Cannot avoid collapse! Exiting...")
            with open(f"{hps.model_dir}/col", 'w') as f:
                f.write(f'{bestEpochStep},{epoch}')
            os._exit(15)
    if rank == 0:
        lastValue = loss_gen_all.item()
    
    if rank == 0 and not hps.if_stop_on_fit:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if rank == 0 and hps.if_stop_on_fit:
        lr = optim_g.param_groups[0]["lr"]
        logger.info(
            f"====> Epoch: {epoch} Step: {global_step} Learning Rate: {lr:.5} {epoch_recorder.record()}"
        )
        # Amor For Tensorboard display
        if loss_mel > 75:
            loss_mel = 75
        if loss_kl > 9:
            loss_kl = 9
        # update tensorboard every epoch
        logger.info(
            f"loss_gen_all={loss_gen_all:.3f}, loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f}, loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
        )
        (image_dict, scalar_dict) = tensorboard_summarize(writer_eval, mel, y_mel, y_hat_mel, loss_disc, losses_disc_r, losses_disc_g, grad_norm_d, loss_mel, loss_kl, loss_fm, losses_gen, loss_gen_all, grad_norm_g, lr)
        dirtyTb.append(
            {
                "global_step": global_step,
                "images": image_dict,
                "scalars": scalar_dict,
            }
        )
        dirtySteps.append(global_step)
        dirtyValues.append(loss_gen_all.item())
        dirtyEpochs.append(epoch)

        best, latest = getBestValue()
        if not continued and epoch - best["epoch"] == 0:
            for i in range(len(dirtyTb)):
                utils.summarize(
                    writer=writer,
                    global_step=dirtyTb[i]["global_step"],
                    images=dirtyTb[i]["images"],
                    scalars=dirtyTb[i]["scalars"],
                )
                if not os.path.exists(f"{hps.model_dir}/fitness.csv"):
                    with open(f"{hps.model_dir}/fitness.csv", 'w', newline='') as f:
                        pass
                with open(f"{hps.model_dir}/fitness.csv", 'a', newline='') as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow([dirtyEpochs[i], dirtySteps[i], dirtyValues[i]])

            dirtyTb.clear()
            dirtySteps.clear()
            dirtyValues.clear()
            dirtyEpochs.clear()
            if epoch > 10:
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_9999999.pth"),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_9999999.pth"),
                )
                bestEpochStep = global_step
                if hasattr(net_g, "module"):
                    ckpt = net_g.module.state_dict()
                else:
                    ckpt = net_g.state_dict()
                logger.info(
                    f'Saving current fittest ckpt: {hps.name}_fittest:{savee(ckpt, hps.sample_rate, hps.if_f0, f"{hps.name}_fittest", epoch, hps.version, hps)}'
                )
        if epoch < 10:
            message = f"Overtrain detection begins in {11 - epoch} epochs"
        elif epoch == 10:
            message = "Overtrain detection will begin next epoch"
        elif epoch - best["epoch"] == 0 and not continued:
            message = f"New best epoch!! [e{epoch}]\n"
        else:
            message = f'Last best epoch [e{best["epoch"]}] seen {epoch - best["epoch"]} epochs ago\n'
        logger.info(message)
        # if epoch - best["epoch"] >= max(len(train_loader), hps.overtrain_epochs):
        change = best["value"] / latest["value"]
        print(change, len(train_loader))
        if change <= 0.95 and latest["epoch"] - best["epoch"] > max(len(train_loader), 20):
            shutil.copy2(f"assets/weights/{hps.name}_fittest.pth", os.path.join(hps.model_dir, f"{hps.name}_{best['epoch']}.pth"))
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_overtrained_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_overtrained_e%s" % (epoch),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )
            shutil.copy2(f"assets/weights/{hps.name + '_overtrained_e%s' % (epoch)}.pth", os.path.join(hps.model_dir,f"{hps.name}_overtrained_e{epoch}.pth"))
            logger.info(
                f'No improvement found after epoch: [e{best["epoch"]}]. The program is closed.'
            )
            os._exit(2333333)

    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                savee(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        if hps.if_stop_on_fit:
            shutil.copy2(f"assets/weights/{hps.name}_fittest.pth", os.path.join(hps.model_dir, f"{hps.name}_{best['epoch']}.pth"))
            shutil.copy2(f"assets/weights/{hps.name}.pth", os.path.join(hps.model_dir,f"{hps.name}_overtrained_e{epoch}.pth"))
        sleep(1)
        os._exit(2333333)
    if continued and rank == 0:
        continued = False

def tensorboard_summarize(writer, mel, y_mel, y_hat_mel, loss_disc, losses_disc_r, losses_disc_g, grad_norm_d, loss_mel, loss_kl, loss_fm, losses_gen, loss_gen_all, grad_norm_g, lr):
    scalar_dict = {
            "loss/g/total": loss_gen_all,
            "loss/d/total": loss_disc,
            "learning_rate": lr,
            "grad_norm_d": grad_norm_d,
            "grad_norm_g": grad_norm_g,
        }
    scalar_dict.update(
            {
                "loss/g/fm": loss_fm,
                "loss/g/mel": loss_mel,
                "loss/g/kl": loss_kl,
            }
        )
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
    scalar_dict.update(
            {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
        )
    scalar_dict.update(
            {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
        )
    image_dict = {
            "slice/mel_org": utils.plot_spectrogram_to_numpy(
                y_mel[0].data.cpu().numpy()
            ),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                y_hat_mel[0].data.cpu().numpy()
            ),
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }
    utils.summarize(
            writer=writer,
            global_step=global_step,
            images=image_dict,
            scalars=scalar_dict,
        )
    return (image_dict, scalar_dict)

def smooth(scalars, weight):
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)
    return smoothed

def getBestValue():
    global lowestValue, dirtySteps, dirtyValues, dirtyEpochs
    steps = []
    values = []
    epochs = []
    if os.path.exists(f"{hps.model_dir}/fitness.csv"):
        with open(f'{hps.model_dir}/fitness.csv', 'r') as f:
            for line in f:
                if line.strip() != "":
                    line = line.split(',')
                    epochs.append(int(line[0]))
                    steps.append(line[1])
                    values.append(float(line[2]))
    steps += dirtySteps
    values = smooth([*values,*dirtyValues], 0.99)
    epochs += dirtyEpochs
    if lowestValue["value"] >= values[-1] and epochs[-1] > 10:
        lowestValue = {"step": steps[-1], "value": values[-1], "epoch": epochs[-1]}
    return [lowestValue, {"step": steps[-1], "value": values[-1], "epoch": epochs[-1]}]



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
