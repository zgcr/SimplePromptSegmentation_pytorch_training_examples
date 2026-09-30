import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp.autocast_mode import autocast


class AverageMeter:
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def all_reduce_operation_in_group_for_variables(variables,
                                                operator,
                                                group=None):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables


def train_universal_segmentation(train_loader, model, criterion, optimizer,
                                 scheduler, epoch, logger, config):
    '''
    train universal segmentation model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader)
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, masks, labels = data['image'], data['mask'], data['label']
        images = images.cuda()

        skip_batch_flag = False

        if torch.any(torch.isinf(images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)):
            skip_batch_flag = True

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                mask_preds, class_preds = model(images)
                loss_value = criterion(mask_preds, class_preds, masks, labels)
        else:
            mask_preds, class_preds = model(images)
            loss_value = criterion(mask_preds, class_preds, masks, labels)

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            print(f'GPU id:{local_rank},zero loss or nan loss or inf loss!')
            skip_batch_flag = True

        loss = loss / config.accumulation_steps
        for key, value in loss_value.items():
            loss_value[key] = value / config.accumulation_steps

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                config.scaler.scale(loss).backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    config.scaler.scale(loss).backward()
        else:
            if iter_index % config.accumulation_steps == 0:
                loss.backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    loss.backward()

        if hasattr(config, 'skip_inf_nan_grad') and config.skip_inf_nan_grad:
            grad_inf_nan_flag = False
            for _, param in model.named_parameters():
                per_weight_grad = param.grad
                if per_weight_grad is not None:
                    if torch.any(torch.isinf(per_weight_grad)) or torch.any(
                            torch.isnan(per_weight_grad)):
                        grad_inf_nan_flag = True
            if grad_inf_nan_flag:
                print(f'GPU id:{local_rank},nan grad or inf grad!')
                skip_batch_flag = True

        [skip_batch_flag] = all_reduce_operation_in_group_for_variables(
            variables=[skip_batch_flag],
            operator=torch.distributed.ReduceOp.SUM)

        if skip_batch_flag:
            log_info = f'skip this batch!'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None
            optimizer.zero_grad()
            continue

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if (hasattr(config, 'clip_grad_value')
                        and config.clip_grad_value
                        > 0) or (hasattr(config, 'clip_max_norm')
                                 and config.clip_max_norm > 0):
                    config.scaler.unscale_(optimizer)

                    if hasattr(config, 'clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), config.clip_grad_value)

                    if hasattr(config, 'clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_max_norm)

                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if (hasattr(config, 'clip_grad_value')
                        and config.clip_grad_value
                        > 0) or (hasattr(config, 'clip_max_norm')
                                 and config.clip_max_norm > 0):

                    if hasattr(config, 'clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), config.clip_grad_value)

                    if hasattr(config, 'clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_max_norm)

                optimizer.step()
                optimizer.zero_grad()

        if config.use_ema_model:
            if iter_index % config.accumulation_steps == 0:
                config.ema_model.update(model)

        if iter_index % config.accumulation_steps == 0:
            for key, value in loss_value.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value], operator=torch.distributed.ReduceOp.SUM)
                loss_value[key] = value / float(config.gpus_num)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss], operator=torch.distributed.ReduceOp.SUM)
            loss = loss / float(config.gpus_num)
            losses.update(loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, total_loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def train_universal_segmentation_deepspeed(train_loader, model, criterion,
                                           optimizer, scheduler, epoch, logger,
                                           config):
    '''
    train universal segmentation model for one epoch using DeepSpeed engine.
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader)
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'
    assert config.accumulation_steps == model.gradient_accumulation_steps()

    for _, data in enumerate(train_loader):
        images, masks, labels = data['image'], data['mask'], data['label']
        images = images.cuda()

        # DeepSpeed engine handles fp16/bf16 casting natively,
        # but criterion is not managed by DeepSpeed, so we use autocast for it
        mask_preds, class_preds = model(images)

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                loss_value = criterion(mask_preds, class_preds, masks, labels)
        else:
            loss_value = criterion(mask_preds, class_preds, masks, labels)

        loss = sum(loss_value.values())

        # DeepSpeed backward (handles loss scaling for fp16 internally)
        model.backward(loss)
        # DeepSpeed step (handles gradient clipping + optimizer step + gradient accumulation internally)
        model.step()

        if config.use_ema_model:
            if iter_index % config.accumulation_steps == 0:
                config.ema_model.update(model)

        if iter_index % config.accumulation_steps == 0:
            for key, value in loss_value.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value], operator=torch.distributed.ReduceOp.SUM)
                loss_value[key] = value / float(config.gpus_num)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss], operator=torch.distributed.ReduceOp.SUM)
            loss = loss / float(config.gpus_num)
            losses.update(loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, total_loss: {loss:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg

    return avg_loss


def train_prompt_segmentation(train_loader, model, criterion, optimizer,
                              scheduler, epoch, logger, config):
    '''
    train prompt segmentation model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader)
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images, masks, pil_images = data['image'], data['mask'], data[
            'pil_image']
        prompt_texts, prompt_languages = data['prompt_text'], data[
            'prompt_language']
        images = images.cuda()
        mask_gts = [m.cuda() for m in masks]
        # class_gts: list of [N_i] tensors from instance_to_prompt_ids.
        # Each GT mask's class label = the prompt index it belongs to.
        # For 1:1 datasets (each prompt → 1 mask), this equals [0, 1, 2, ...].
        # For 1:M datasets (each prompt → M instance masks), multiple masks
        # share the same class label (prompt index), enabling the model to
        # predict multiple instances per prompt via Hungarian matching.
        class_gts = [ids.cuda() for ids in data['instance_to_prompt_ids']]

        sample_type = data['sample_type']
        vprompt_masks = None
        prompt_type_id = None
        if sample_type == 'visual':
            vprompt_masks = [m.cuda() for m in data['prompt_mask']]
            # Tells the region encoder whether the prompt came from a point,
            # a box or a mask, which are otherwise indistinguishable once the
            # collater has rendered all three into the same binary mask.
            prompt_type_id = data['visual_prompt_type_id'].cuda()

        # Non-padded (height, width) of each image, used to normalise the
        # region geometry encoding so that letterbox padding does not bias it.
        valid_sizes = torch.as_tensor(data['size']).float().cuda()

        skip_batch_flag = False

        if torch.any(torch.isinf(images)):
            skip_batch_flag = True

        if torch.any(torch.isnan(images)):
            skip_batch_flag = True

        # ---- Tokenize ----
        # The tokenizer handles template selection, chat message building,
        # and tokenization in one call.
        tokenized = config.tokenizer.encode(prompt_texts=prompt_texts,
                                            sample_type=sample_type,
                                            prompt_languages=prompt_languages,
                                            pil_images=pil_images)

        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        labels = tokenized['labels'].cuda()
        cond_ids = tokenized['cond_ids'].cuda()
        seg_ids = tokenized['seg_ids'].cuda()
        pixel_values = tokenized['pixel_values']
        image_grid_thw = tokenized['image_grid_thw']
        mm_token_type_ids = tokenized['mm_token_type_ids']

        pixel_values = pixel_values.cuda()
        image_grid_thw = image_grid_thw.cuda()
        mm_token_type_ids = mm_token_type_ids.cuda()

        # The tokenizer emits one <SEG> and one <p>...</p> span per prompt,
        # and the collater derives class_gts from the same prompt ordering.
        # Verifying the counts here turns a silent misalignment (which would
        # only show up as poor convergence) into an immediate failure.
        #
        # The reference is the number of DISTINCT prompt ids, not the number
        # of GT masks. Both agree for a 1:1 dataset (instance_to_prompt_ids is
        # then arange(n)), but for a 1:M dataset several masks share one prompt
        # id, so comparing against the mask count would reject a perfectly
        # aligned batch.
        num_prompts = tokenized['num_prompts']
        num_unique_prompts = torch.as_tensor(
            [int(per_ids.unique().numel()) for per_ids in class_gts],
            device=num_prompts.device)
        assert torch.equal(num_prompts, num_unique_prompts), (
            f'prompt count {num_prompts.tolist()} does not match unique '
            f'prompt id count {num_unique_prompts.tolist()}')

        # ---- Model forward ----
        # region_token_id lets the model find the <region> placeholders that
        # the tokenizer emitted, so region features can be scattered onto them
        region_token_id = config.tokenizer.region_token_id

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                mask_preds, class_preds, vlm_loss = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    cond_ids=cond_ids,
                    seg_ids=seg_ids,
                    vprompt_masks=vprompt_masks,
                    labels=labels,
                    region_token_id=region_token_id,
                    valid_sizes=valid_sizes,
                    prompt_type_id=prompt_type_id)
                loss_value = criterion(
                    mask_preds,
                    class_preds,
                    mask_gts,
                    class_gts,
                    vlm_loss=vlm_loss,
                )
        else:
            mask_preds, class_preds, vlm_loss = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cond_ids=cond_ids,
                seg_ids=seg_ids,
                vprompt_masks=vprompt_masks,
                labels=labels,
                region_token_id=region_token_id,
                valid_sizes=valid_sizes,
                prompt_type_id=prompt_type_id,
            )
            loss_value = criterion(
                mask_preds,
                class_preds,
                mask_gts,
                class_gts,
                vlm_loss=vlm_loss,
            )

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for key, value in loss_value.items():
            if torch.any(torch.isinf(value)) or torch.any(torch.isnan(value)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            print(f'GPU id:{local_rank},zero loss or nan loss or inf loss!')
            skip_batch_flag = True

        loss = loss / config.accumulation_steps
        for key, value in loss_value.items():
            loss_value[key] = value / config.accumulation_steps

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                config.scaler.scale(loss).backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    config.scaler.scale(loss).backward()
        else:
            if iter_index % config.accumulation_steps == 0:
                loss.backward()
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    loss.backward()

        if hasattr(config, 'skip_inf_nan_grad') and config.skip_inf_nan_grad:
            grad_inf_nan_flag = False
            for _, param in model.named_parameters():
                per_weight_grad = param.grad
                if per_weight_grad is not None:
                    if torch.any(torch.isinf(per_weight_grad)) or torch.any(
                            torch.isnan(per_weight_grad)):
                        grad_inf_nan_flag = True
            if grad_inf_nan_flag:
                print(f'GPU id:{local_rank},nan grad or inf grad!')
                skip_batch_flag = True

        [skip_batch_flag] = all_reduce_operation_in_group_for_variables(
            variables=[skip_batch_flag],
            operator=torch.distributed.ReduceOp.SUM)

        if skip_batch_flag:
            log_info = f'skip this batch!'
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None
            optimizer.zero_grad()
            continue

        if config.use_amp:
            if iter_index % config.accumulation_steps == 0:
                if (hasattr(config, 'clip_grad_value')
                        and config.clip_grad_value
                        > 0) or (hasattr(config, 'clip_max_norm')
                                 and config.clip_max_norm > 0):
                    config.scaler.unscale_(optimizer)

                    if hasattr(config, 'clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), config.clip_grad_value)

                    if hasattr(config, 'clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_max_norm)

                config.scaler.step(optimizer)
                config.scaler.update()
                optimizer.zero_grad()
        else:
            if iter_index % config.accumulation_steps == 0:
                if (hasattr(config, 'clip_grad_value')
                        and config.clip_grad_value
                        > 0) or (hasattr(config, 'clip_max_norm')
                                 and config.clip_max_norm > 0):

                    if hasattr(config, 'clip_grad_value'):
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), config.clip_grad_value)

                    if hasattr(config, 'clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.clip_max_norm)

                optimizer.step()
                optimizer.zero_grad()

        if iter_index % config.accumulation_steps == 0:
            for key, value in loss_value.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value], operator=torch.distributed.ReduceOp.SUM)
                loss_value[key] = value / float(config.gpus_num)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss], operator=torch.distributed.ReduceOp.SUM)
            loss = loss / float(config.gpus_num)
            losses.update(loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, total_loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def train_prompt_segmentation_deepspeed(train_loader, model, criterion,
                                        optimizer, scheduler, epoch, logger,
                                        config):
    '''
    train prompt segmentation model for one epoch using DeepSpeed engine.
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader)
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'
    assert config.accumulation_steps == model.gradient_accumulation_steps()

    for _, data in enumerate(train_loader):
        images, masks, pil_images = data['image'], data['mask'], data[
            'pil_image']
        prompt_texts, prompt_languages = data['prompt_text'], data[
            'prompt_language']
        images = images.cuda()
        mask_gts = [m.cuda() for m in masks]
        # class_gts: list of [N_i] tensors from instance_to_prompt_ids.
        # Each GT mask's class label = the prompt index it belongs to.
        # For 1:1 datasets (each prompt → 1 mask), this equals [0, 1, 2, ...].
        # For 1:M datasets (each prompt → M instance masks), multiple masks
        # share the same class label (prompt index), enabling the model to
        # predict multiple instances per prompt via Hungarian matching.
        class_gts = [ids.cuda() for ids in data['instance_to_prompt_ids']]

        sample_type = data['sample_type']
        vprompt_masks = None
        prompt_type_id = None
        if sample_type == 'visual':
            vprompt_masks = [m.cuda() for m in data['prompt_mask']]
            # Tells the region encoder whether the prompt came from a point,
            # a box or a mask, which are otherwise indistinguishable once the
            # collater has rendered all three into the same binary mask.
            prompt_type_id = data['visual_prompt_type_id'].cuda()

        # Non-padded (height, width) of each image, used to normalise the
        # region geometry encoding so that letterbox padding does not bias it.
        valid_sizes = torch.as_tensor(data['size']).float().cuda()

        # ---- Tokenize ----
        # The tokenizer handles template selection, chat message building,
        # and tokenization in one call.
        tokenized = config.tokenizer.encode(prompt_texts=prompt_texts,
                                            sample_type=sample_type,
                                            prompt_languages=prompt_languages,
                                            pil_images=pil_images)

        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        labels = tokenized['labels'].cuda()
        cond_ids = tokenized['cond_ids'].cuda()
        seg_ids = tokenized['seg_ids'].cuda()
        pixel_values = tokenized['pixel_values']
        image_grid_thw = tokenized['image_grid_thw']
        mm_token_type_ids = tokenized['mm_token_type_ids']

        pixel_values = pixel_values.cuda()
        image_grid_thw = image_grid_thw.cuda()
        mm_token_type_ids = mm_token_type_ids.cuda()

        # The tokenizer emits one <SEG> and one <p>...</p> span per prompt,
        # and the collater derives class_gts from the same prompt ordering.
        # Verifying the counts here turns a silent misalignment (which would
        # only show up as poor convergence) into an immediate failure.
        #
        # The reference is the number of DISTINCT prompt ids, not the number
        # of GT masks. Both agree for a 1:1 dataset (instance_to_prompt_ids is
        # then arange(n)), but for a 1:M dataset several masks share one prompt
        # id, so comparing against the mask count would reject a perfectly
        # aligned batch.
        num_prompts = tokenized['num_prompts']
        num_unique_prompts = torch.as_tensor(
            [int(per_ids.unique().numel()) for per_ids in class_gts],
            device=num_prompts.device)
        assert torch.equal(num_prompts, num_unique_prompts), (
            f'prompt count {num_prompts.tolist()} does not match unique '
            f'prompt id count {num_unique_prompts.tolist()}')

        # ---- Model forward ----
        # region_token_id lets the model find the <region> placeholders that
        # the tokenizer emitted, so region features can be scattered onto them
        region_token_id = config.tokenizer.region_token_id

        # DeepSpeed engine handles fp16/bf16 casting natively,
        # but criterion is not managed by DeepSpeed, so we use autocast for it
        mask_preds, class_preds, vlm_loss = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cond_ids=cond_ids,
            seg_ids=seg_ids,
            vprompt_masks=vprompt_masks,
            labels=labels,
            region_token_id=region_token_id,
            valid_sizes=valid_sizes,
            prompt_type_id=prompt_type_id)

        if config.use_amp:
            with autocast(device_type="cuda", dtype=config.amp_type):
                loss_value = criterion(
                    mask_preds,
                    class_preds,
                    mask_gts,
                    class_gts,
                    vlm_loss=vlm_loss,
                )
        else:
            loss_value = criterion(
                mask_preds,
                class_preds,
                mask_gts,
                class_gts,
                vlm_loss=vlm_loss,
            )

        loss = sum(loss_value.values())

        # DeepSpeed backward (handles loss scaling for fp16 internally)
        model.backward(loss)
        # DeepSpeed step (handles gradient clipping + optimizer step + gradient accumulation internally)
        model.step()

        if iter_index % config.accumulation_steps == 0:
            for key, value in loss_value.items():
                [value] = all_reduce_operation_in_group_for_variables(
                    variables=[value], operator=torch.distributed.ReduceOp.SUM)
                loss_value[key] = value / float(config.gpus_num)

            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss], operator=torch.distributed.ReduceOp.SUM)
            loss = loss / float(config.gpus_num)
            losses.update(loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, total_loss: {loss:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg

    return avg_loss
