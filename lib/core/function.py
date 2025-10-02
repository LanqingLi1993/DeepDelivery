import _init_paths
from .evaluate import accuracy, AverageMeter, ConfusionMatrix, Evaluator
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.distributed as dist
import time
from tqdm import tqdm
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    pass
import os
from metadata import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def train_model(
        trainLoader, model, epoch, epoch_number, optimizer, combiner, criterion, cfg, logger, rank=0, use_apex=True, **kwargs
):
    if cfg['eval_mode']:
        model.eval()
    else:
        model.train()

    trainLoader.dataset.update(epoch)
    combiner.update(epoch)
    criterion.update(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    labels = []
    preds = []
    for i, batch_dic in enumerate(trainLoader):
        data = batch_dic['x']
        label = batch_dic['y']
        meta = batch_dic['meta']
        meta_data, meta_label = batch_dic['meta_data'], batch_dic['meta_label']
        lds_weight = batch_dic['lds_weight'] if batch_dic.__contains__('lds_weight') else None
        cnt = label.shape[0]
        loss, now_acc, label, pred = combiner.forward(model, criterion, data, label,
                                                      meta, meta_data, meta_label, lds_weight=lds_weight, epoch=epoch, training=True)  # feature-level mix

        optimizer.zero_grad()

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)
        labels += list(label)
        preds += list(pred)

        if i % cfg['show_step'] == 0 and rank == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)

    end_time = time.time()

    # statistics for long-tailed training metrics
    pbar_str = "---Epoch:{:>3d}/{}".format(epoch, epoch_number)
    lt_metrics = metrics[cfg['setting']['type']]
    for lt_metric in lt_metrics:
        lt_evaluator = Evaluator(name=lt_metric)
        if cfg['setting']['type'] == 'LT Regression':
            # result = lt_evaluator(labels, preds, sample_weight=label_weights)
            result = lt_evaluator(labels, preds)
        elif lt_metric not in ['roc-auc']:
            result = lt_evaluator(labels, preds)
        else: 
            continue
        pbar_str += "   {}:{:>5.4f}".format(lt_metric, result)
    if rank == 0:
        logger.info(pbar_str)

    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc.avg, all_loss.avg

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def valid_model(
        dataLoader, epoch_number, model, cfg, criterion, logger, device, rank, distributed, **kwargs
):
    model.eval()

    if cfg['loss']['type']=="DiVEKLD":
        criterion = criterion.base_loss
    with torch.no_grad():
        all_loss = AverageMeter()
        acc_avg = AverageMeter()

        labels = []
        preds = []
        pred_scores = []
        label_weights = []
        now_results = []

        func = torch.nn.Sigmoid() \
            if cfg['loss']['type'] in ['FocalLoss', 'ClassBalancedFocal'] else \
            torch.nn.Softmax(dim=1)

        for i, batch_dic in enumerate(dataLoader):
            data = batch_dic['x']
            label = batch_dic['y']
            label_weights += batch_dic['y_weight']
            data, label = data.to(device), label.to(device)
            feature, att_mats = model(data, feature_flag=True)

            output = model(feature, head_flag=True, label=label)

            labels += list(label.cpu().numpy())
            preds += list(torch.argmax(output, 1).cpu().numpy())
            pred_scores += list(func(output).cpu().numpy())

            loss = criterion(output, label.long(), feature=feature)

            if cfg['setting']['type'] == "LT Regression":
                now_result = output[:, 0]
            elif cfg['setting']['type'] in ["LT Classification","Open LT", "Imbalanced Learning"]:
                score_result = func(output)
                now_result = torch.argmax(score_result, 1)
            now_results += list(now_result.cpu().numpy())
            acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())

            if distributed:
                world_size = float(os.environ.get("WORLD_SIZE", 1))
                reduced_loss = reduce_tensor(loss.data, world_size)
                reduced_acc = reduce_tensor(torch.from_numpy(np.array([acc])).cuda(), world_size)
                loss = reduced_loss.cpu().data
                acc = reduced_acc.cpu().data

            all_loss.update(loss.data.item(), label.shape[0])
            if distributed:
                acc_avg.update(acc.data.item(), cnt*world_size)
            else:
                acc_avg.update(acc, cnt)

        # statistics for long-tailed validation metrics
        pbar_str = "------- Valid: Epoch:{:>3d}".format(epoch_number)
        lt_metrics = metrics[cfg['setting']['type']]
        lt_results = []
        for lt_metric in lt_metrics:
            lt_evaluator = Evaluator(name=lt_metric)
            if cfg['setting']['type'] == 'LT Regression':
                result = lt_evaluator(labels, now_results, sample_weight=label_weights)
            elif lt_metric in ['roc-auc']:
                result = lt_evaluator(labels, pred_scores)
            else:
                result = lt_evaluator(labels, preds)
            lt_results.append(result)
            pbar_str += "  {}:{:>5.4f}".format(lt_metric, result)
        if rank == 0:
            logger.info(pbar_str)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f} Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc_avg.avg * 100
        )
        if rank == 0:
            logger.info(pbar_str)
    return acc_avg.avg, all_loss.avg, lt_results, lt_metrics

def test_model(
        dataLoader, model, cfg, logger, device, output_df=True, **kwargs
):
    model.eval()
    pbar = tqdm(total=len(dataLoader))

    all_labels = []
    all_preds = []
    all_pred_regs = []
    all_pred_scores = []
    n_batch = 0

    if cfg['test']['save_format'] == 'pkl':
        df_pkl = pd.DataFrame()
    with torch.no_grad():
        all_loss = AverageMeter()
        acc_avg = AverageMeter()

        if cfg['setting']['type'] == 'Open LT':
           lt_cls_dic = {'head': [], 'middle': [], 'tail': [], 'open': []}
        else:
           lt_cls_dic = {'head': [], 'middle': [], 'tail': []}

        func = torch.nn.Sigmoid() \
            if cfg['loss']['type'] in ['FocalLoss', 'ClassBalancedFocal'] else \
            torch.nn.Softmax(dim=1)

        for batch_dic in dataLoader:
            if output_df:
                x_raw = []
                att_map = {}
            labels = []
            pred_scores = []
            preds = []
            pred_regs = []
            label_weights = []
            ids = []
            x_raw += list(zip(*batch_dic['x_raw']))
            #print(i)
            #print(batch_dic)
            id = batch_dic['x_id']
            ids += list(zip(*batch_dic['x_id']))
            data = batch_dic['x']
            label = batch_dic['y']
            lt_labels = batch_dic['lt_class']
            if cfg['setting']['type'] not in ['LT Regression']:
                for i, cls_label in enumerate(label):
                    lt_label = lt_labels[i]
                    lt_label = ''.join([i for i in lt_label if not i.isdigit()]) # get one of ['head', 'middle', 'tail', 'open']
                    if lt_label == 'head:':
                        lt_label = lt_label.replace('head:', 'head')
                    if cls_label not in lt_cls_dic[lt_label]:
                        lt_cls_dic[lt_label].append(cls_label)

            label_weights += batch_dic['y_weight']
            data, label = data.to(device), label.to(device)
            feature, att_mats = model(data, feature_flag=True) # att_mats consists of a list of attention tensor for each backbone model, where the size of each attention tensor is (num_layer, n_batch, seq_len)
            for i, att_mat in enumerate(att_mats):
                for j, att_mat_layer in enumerate(att_mat):
                    if '%s_layer%d' %(model.backbone_kws[i], j) not in att_map:
                        att_map['%s_layer%d' %(model.backbone_kws[i], j)] = []
                    att_map['%s_layer%d' %(model.backbone_kws[i], j)] += list(att_mat_layer.cpu().numpy())
            
            output = model(feature, head_flag=True, label=label)
            output_0 = output[:,0]
            labels += list(label.cpu().numpy())
            all_labels += labels
            if cfg['setting']['type'] == 'Open LT':
                temp_ss = func(output).cpu().numpy()
                if cfg['baseline'] == 'OLTR':
                    clf = IsolationForest(random_state=1).fit(output.cpu())
                    ano = clf.predict(output.cpu())
                    class_open_sm = -ano*np.ones(temp_ss.shape[0])
                    class_open_sm = class_open_sm[:,np.newaxis]
                    temp_ss = np.concatenate((temp_ss, class_open_sm), axis=1)
                elif cfg['baseline'] == 'IEM':
                    clf = LocalOutlierFactor(n_neighbors=10)
                    ano = clf.fit_predict(output.cpu())
                    class_open_sm = -ano*np.ones(temp_ss.shape[0])
                    class_open_sm = class_open_sm[:,np.newaxis]
                    temp_ss = np.concatenate((temp_ss, class_open_sm), axis=1)

                            # if use_IsolationForest:
                #     clf = IsolationForest(random_state=0).fit(output.cpu())
                #     ano = clf.predict(output.cpu())
                #     class_open_sm = ano*np.ones(temp_ss.shape[0])
                #     class_open_sm = class_open_sm[:,np.newaxis]
                #     temp_ss = np.concatenate((temp_ss, class_open_sm), axis=1)

                else:
                    temp_ss = np.concatenate((temp_ss, 0.2*np.ones((temp_ss.shape[0],1))), axis=1)
                pred_scores += list(temp_ss)
            else:
                pred_scores += list(func(output).cpu().numpy())
            all_pred_scores += pred_scores
            preds = list(torch.argmax(torch.tensor(pred_scores), 1).cpu().numpy())
            all_preds += preds
            # preds += list(torch.argmax(output, 1).cpu().numpy())
            pred_regs += list(output_0.cpu().numpy())
            all_pred_regs += pred_regs

            # loss = criterion(output, label, feature=feature)
            # score_result = func(output)

            # now_result = torch.argmax(score_result, 1)
            acc, cnt = accuracy(np.array(preds), np.array(labels))
            pbar.set_description("Now Acc:{:>5.2f}%".format(acc * 100))
            pbar.update(1)

            
            # write output to .csv file once per batch
            if cfg["network"]["pretrained"]: # generate embeddings from pretrained model
                df = pd.DataFrame({'ID': ids, 'x_raw': x_raw,  'Y': labels, 'y_pred': preds, 'y_pred_scores': pred_scores})
                for key, value in att_map.items():
                    # maximum embedding dim of ESM models, otherwise not all entries of the embedding vector will be stored when converted to string
                    np.set_printoptions(threshold = 5120) 
                    df["X"] = value
                df['x_raw'] = df['x_raw'].apply(post_processing)
                df['ID'] = df['ID'].apply(post_processing)
            else:
                df = pd.DataFrame({'ID': ids, 'x_raw': x_raw, 'y': labels, 'y_pred': preds, 'y_pred_scores': pred_scores})
                for key, value in att_map.items():
                    df[key] = value
                df['x_raw'] = df['x_raw'].apply(post_processing)
                df['ID'] = df['ID'].apply(post_processing)


            if cfg['test']['save_format'] == 'csv':
                output_csv = logger.handlers[0].baseFilename.replace('.log', '.csv')
                if n_batch == 0:
                    df.to_csv(output_csv, index=False)
                else:
                    df.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False) 
                n_batch += 1
            elif cfg['test']['save_format'] == 'pkl':    
                df_pkl = df_pkl.append(df)

        if cfg['test']['save_format'] == 'pkl':    
            output_pkl = logger.handlers[0].baseFilename.replace('.log', '.pkl')
            df_pkl.to_pickle(output_pkl)
            # if distributed:
            #     world_size = float(os.environ.get("WORLD_SIZE", 1))
            #     reduced_loss = reduce_tensor(loss.data, world_size)
            #     reduced_acc = reduce_tensor(torch.from_numpy(np.array([acc])).cuda(), world_size)
            #     loss = reduced_loss.cpu().data
            #     acc = reduced_acc.cpu().data

            # all_loss.update(loss.data.item(), label.shape[0])
            # if distributed:
            #     acc_avg.update(acc.data.item(), cnt*world_size)
            # else:
            #     acc_avg.update(acc, cnt)

        

        # statistics for long-tailed validation metrics
        lt_metrics = metrics[cfg['setting']['type']]
        lt_results = []
        for lt_metric in lt_metrics:
            pbar_str = "------- Test: "
            lt_evaluator = Evaluator(name=lt_metric)
            if cfg['setting']['type'] == 'LT Regression':
                result = lt_evaluator(all_labels, all_pred_regs, sample_weight=label_weights)
                pbar_str += "  {}:{:>5.4f}".format(lt_metric, result) 
                for lt_cls, lt_idx in lt_cls_dic.items():
                    if len(lt_idx) == 0:
                        pbar_str += "  {}_{}:N/A".format(lt_metric, lt_cls)
                    else: 
                        sample_weight = label_weights * np.isin(all_labels, lt_idx)
                        result = lt_evaluator(all_labels, all_preds, sample_weight=sample_weights)
                        pbar_str += "  {}_{}:{:>5.4f}".format(lt_metric, lt_cls, result)

            else:
                if lt_metric in ['balanced_accuracy', 'balanced-f1']:
                    result = lt_evaluator(all_labels, all_preds)
                    pbar_str += "  {}:{:>5.4f}".format(lt_metric, result) 
                    result_per_cls = lt_evaluator(all_labels, all_preds, per_class=True)
                    cls_labels = np.arange(result_per_cls.size)
                    for lt_cls, lt_idx in lt_cls_dic.items():
                        if len(lt_idx) == 0:
                            pbar_str += "  {}_{}: N/A".format(lt_metric, lt_cls)
                        else: 
                            pbar_str += "  {}_{}:{:>5.4f}".format(lt_metric, lt_cls, np.average(result_per_cls, weights=np.isin(cls_labels, lt_idx)+1e-12)) # avoid zero division
                elif lt_metric in ['roc-auc']:
                    print('lt_cls_dic', lt_cls_dic)
                    # labels = np.array(labels)
                    # pred_scores = np.array(pred_scores)
                    # onehot_labels = np.zeros((labels.size, int(labels.max()+1)))
                    # onehot_labels[np.arange(labels.size), labels.astype('int64')] = 1
                    result_per_cls = lt_evaluator(all_labels, all_pred_scores, per_class=True)
                    cls_labels = np.arange(result_per_cls.size)
                    if isinstance(result_per_cls, float):
                        pbar_str += "  {}:{:>5.4f}".format(lt_metric, result_per_cls)
                    else:
                        pbar_str += "  {}:{:>5.4f}".format(lt_metric, np.average(result_per_cls))
                        for lt_cls, lt_idx in lt_cls_dic.items():
                            if len(lt_idx) == 0:
                                pbar_str += "  {}_{}: N/A".format(lt_metric, lt_cls)
                            else: 
                                pbar_str += "  {}_{}:{:>5.4f}".format(lt_metric, lt_cls, np.average(result_per_cls, weights=np.isin(cls_labels, lt_idx)))
                else:
                    result = lt_evaluator(all_labels, all_preds)
                    pbar_str += "  {}:{:>5.4f}".format(lt_metric, result) 
                
            lt_results.append(result)

            logger.info(pbar_str)


        m = ConfusionMatrix(len(set(all_labels)))
        m.update(np.array(all_labels), np.array(all_preds))
        m.plot_confusion_matrix()
        plt.savefig(logger.handlers[0].baseFilename.replace('.log', '.pdf'))
            

    return lt_results, lt_metrics


def post_processing(s):
    return str(s)[2:-3]
    

