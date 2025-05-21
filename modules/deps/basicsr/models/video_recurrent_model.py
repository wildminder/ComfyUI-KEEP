import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import cv2
import os

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class VideoRecurrentModel(SRModel):
    """Video Recurrent SR model (merged with VideoBaseModel)."""

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(
            f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if hasattr(self, 'fix_flow_iter') and self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(
                    f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        super(VideoRecurrentModel, self).optimize_parameters(current_iter)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        save_video = self.opt['val'].get('save_video', False)
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if hasattr(self, 'center_frame_only') and self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # # For EDVR
            # result = visuals['result']
            # result_img = tensor2img([result])

            # if save_img:
            #     if self.opt['is_train']:
            #         raise NotImplementedError(
            #             'saving image is not supported during training.')
            #     else:
            #         img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
            #                             f"{idx:08d}.png")
            #         # image name only for REDS dataset
            #     imwrite(result_img, img_path)

            # evaluate
            if i < num_folders:
                video_writer = None
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img(
                        [result], min_max=(-1, 1))  # uint8, bgr
                    metric_data['img1'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img(
                            [gt], min_max=(-1, 1))  # uint8, bgr
                        metric_data['img2'] = gt_img

                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError(
                                'saving image is not supported during training.')
                        else:
                            if hasattr(self, 'center_frame_only') and self.center_frame_only:  # vimeo-90k
                                clip_ = val_data['lq_path'].split('/')[-3]
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{name_}_{self.opt['name']}.png")
                            else:  # others
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{idx:08d}.png")
                        imwrite(result_img, img_path)

                    if save_video:
                        if self.opt['is_train']:
                            raise NotImplementedError(
                                'saving image is not supported during training.')
                        else:
                            if video_writer is None:
                                video_output_path = osp.join(self.opt['path']['visualization'], dataset_name+'_video',
                                                             f"{folder}.mp4")
                                dir_name = osp.abspath(
                                    osp.dirname(video_output_path))
                                os.makedirs(dir_name, exist_ok=True)
                                frame_rate = 15
                                h, w = result_img.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                video_writer = cv2.VideoWriter(video_output_path, fourcc,
                                                               frame_rate, (w, h))
                            video_writer.write(result_img)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx,
                                                        metric_idx] += result

                if save_video:
                    cv2.destroyAllWindows()
                    video_writer.release()

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(
                    current_iter, dataset_name, tb_logger)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.warning(
            'nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(dataloader, current_iter, tb_logger, save_img)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        # ----------------- calculate the average values for each folder, and for each metric  ----------------- #
        # average all frames for each sub-folder
        # metric_results_avg is a dict:{
        #    'folder1': tensor (len(metrics)),
        #    'folder2': tensor (len(metrics))
        # }
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }
        # total_avg_results is a dict: {
        #    'metric1': float,
        #    'metric2': float
        # }
        total_avg_results = {
            metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()
        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)
            # update the best metric result
            self._update_best_metric_result(
                dataset_name, metric, total_avg_results[metric], current_iter)

        # ------------------------------------------ log the metric ------------------------------------------ #
        log_str = f'Validation {dataset_name}\n'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}\n'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}\n'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\n\t    Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(
                        f'metrics/{metric}/{folder}', tensor[metric_idx].item(), current_iter)

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            video_length = self.lq.shape[1]
            fix_length = 20
            if video_length > fix_length:
                output = []
                for start_idx in range(0, video_length, fix_length):
                    end_idx = min(start_idx + fix_length, video_length)
                    if end_idx - start_idx == 1:
                        output.append(self.net_g(
                            self.lq[:, [start_idx, start_idx], ...])[:, 0:1, ...])
                    else:
                        output.append(self.net_g(
                            self.lq[:, start_idx:end_idx, ...]))
                self.output = torch.cat(output, dim=1)
                assert self.output.shape[1] == video_length, "Differer number of frames"
            else:
                self.output = self.net_g(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if hasattr(self, 'center_frame_only') and self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
