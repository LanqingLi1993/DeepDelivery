
import os
import torch
import shutil
import logging
import numpy as np
from net import Network
from datetime import datetime
from os import path as osp
import logging

from DeepPurpose.utils import generate_config
import DeepPurpose.CompoundPred as single_pred
from DeepPurpose.utils import idx2word_d, idx2word_p

from .lr_scheduler import WarmupMultiStepLR
from dataset.base_dataset import Dataset
import torch.nn.functional as F
from transformers import EsmModel

logger = logging.getLogger(__name__)

def deep_update_dict(fr, to):
    """ update dict of dicts with new values """
    # assume dicts have same keys
    for k, v in fr.items():
        if isinstance(v, dict):
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def create_logger(cfg, rank=0, test=False):
    dataset = cfg["dataset"]["dataset_name"]
    drug_encoding = cfg["dataset"]["drug_encoding"]
    protein_encoding = cfg["dataset"]["protein_encoding"]
    head_type = cfg["head"]["type"]
    if test: # for testing
        log_dir = osp.join(cfg["output_dir"], dataset, "test")

        if cfg["debug"]:
            log_name = "debug.log"
        else:
            log_name = "{}.log".format(cfg["test"]["exp_id"])

    else:
        log_dir = osp.join(cfg["output_dir"], dataset, "logs")

        if cfg["debug"]:
            log_name = "debug.log"
        else:
            log_dir = osp.join(cfg["output_dir"], dataset, "logs")
            time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            # log_name = "{}_{}_{}_{}_{}.log".format(dataset, drug_encoding, protein_encoding, head_type, time_str)

            loss = cfg["loss"]["type"]
            seed = cfg["seed"]
            log_name = "{}_{}_{}_{}_{}_{}.log".format(dataset, drug_encoding, loss, seed, head_type, time_str)

    log_file = osp.join(log_dir, log_name)
    if not osp.exists(log_dir) and rank == 0:
        os.makedirs(log_dir)

    # set up logger
    print("=> creating log {}".format(log_file))
    header = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=header, force=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if rank > 0:
        return logger, log_file
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file, log_name.split(".")[0]

def get_optimizer(cfg, model):
    lr = cfg["train"]["optimizer"]["lr"]
    optim_type = cfg["train"]["optimizer"]["type"]
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})
    if optim_type == "SGD":
        optimizer = torch.optim.SGD(
            params=params,
            lr=lr,
            momentum=cfg["train"]["optimizer"]["momentum"],
            weight_decay=cfg["train"]["optimizer"]["wc"],
            nesterov=True,
        )
    elif optim_type == "ADAM":
        optimizer = torch.optim.Adam(
            params=params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=cfg["train"]["optimizer"]["wc"],
        )
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(cfg, optimizer):
    if cfg["train"]["lr_scheduler"]["type"] == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg["train"]["lr_scheduler"]["lr_step"],
            gamma=cfg["train"]["lr_scheduler"]["lr_factor"],
        )
    elif cfg["train"]["lr_scheduler"]["type"] == "cosine":
        if cfg["train"]["lr_scheduler"]["cosine_decay_end"] > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cfg["train"]["lr_scheduler"]["cosine_decay_end"],
                eta_min=1e-4,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cfg["train"]["max_epoch"],
                eta_min=1e-4,
            )
    elif cfg["train"]["lr_scheduler"]["type"] == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=cfg["train"]["lr_scheduler"]["lr_step"],
            gamma=cfg["train"]["lr_scheduler"]["lr_factor"],
            warmup_epochs=cfg["train"]["lr_scheduler"]["warmup_epoch"],
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg["train"]["lr_scheduler"]["type"]))

    return scheduler

def get_model(cfg, device, logger, entity_type: list):
    drug_encoding = cfg["dataset"]["drug_encoding"]
    protein_encoding = cfg["dataset"]["protein_encoding"]
    config = generate_config(drug_encoding=drug_encoding,
                            target_encoding=protein_encoding,
                            **cfg["backbone"]["deeppurpose"])
    # # Load Huggingface model
    # if cfg["network"]["pretrained"]:
    #     encoding = cfg["dataset"]["{}_encoding".format(entity_type[0])]
    #     if encoding == "ESM":
    #         from transformers import EsmModel
    #         model = EsmModel.from_pretrained(cfg["network"]["pretrained_model"])
    #     else:
    #         raise NotImplementedError

    # else:
    if len(entity_type) == 1: # single-instance prediction
        if entity_type[0] == "drug":
            from DeepPurpose.CompoundPred import model_initialize
        elif entity_type[0] == "protein":
            # from DeepPurpose.ProteinPred import model_initialize
            from net.model_helper import model_initialize
        else:
            raise NotImplementedError
    elif len(entity_type) == 2:
        if "drug" in entity_type and "protein" in entity_type: # drug-target interaction
            from DeepPurpose.DTI import model_initialize
        elif "drug" in entity_type and len(set(entity_type)) == 1: # drug-drug interaction
            from DeepPurpose.DDI import model_initialize
        elif "protein" in entity_type and len(set(entity_type)) == 1: # protein-protein interaction
            from DeepPurpose.PPI import model_initialize
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    model = model_initialize(**config)

    model = Network(model=model, cfg=cfg, mode="train", num_class=cfg["setting"]["num_class"], setting_type=cfg["setting"]["type"], entity_type=entity_type)

    if cfg["backbone"]["freeze"]:
        model.freeze_backbone()
        logger.info("Backbone has been freezed")
    
    model.to(device)

    return model

def get_dataset(cfg):
    if cfg["dataset"]["tier1_task"] == "single_pred":
        from dataset.single_pred import ADME, QM, TestSinglePred, Tox, Yields, ReactType, Transposition, BioAct
    elif cfg["dataset"]["tier1_task"] == "multi_pred":
        from dataset.multi_pred import Catalyst, DDI, DTI, PPI, TestMultiPred, ReactType, Yields
    elif cfg["dataset"]["tier1_task"] == "generation":
        from dataset.generation import MolGen, Reaction, RetroSyn
    else:
        raise NotImplementedError

    # Handle tasks with multiple labels
    # try:

    dataset = eval(cfg["dataset"]["tier2_task"])(name = cfg["dataset"]["dataset_name"], \
        path = cfg["dataset"]["path"])
    # except:
    #     print("This is a multi-label task.")
    #     from utils import retrieve_label_name_list
    #     name_list = retrieve_label_name_list(cfg["dataset"]["dataset_name"])
    #     label_name = cfg["dataset"]["split"]["label_name"]
    #     if cfg["dataset"]["split"]["label_name"] in name_list:
    #         dataset = eval(cfg["dataset"]["tier2_task"])(name = cfg["dataset"]["dataset_name"], \
    #             label_name=label_name, path = cfg["dataset"]["path"])
    #     else:
    #         dataset = eval(cfg["dataset"]["tier2_task"])(name = cfg["dataset"]["dataset_name"], \
    #             label_name=name_list[0], path = cfg["dataset"]["path"])

    if cfg["setting"]["type"] == "LT Regression":
        label_type = "regression"
    elif cfg["setting"]["type"] in ["LT Classification", "Open LT", "Imbalanced Learning"]:
        label_type = "classification"
    else:
        raise NotImplementedError

    num_class = cfg["setting"]["num_class"]

    if cfg["setting"]["type"] == "Open LT":
        cfg["dataset"]["split"]["method"] = "open-" + cfg["dataset"]["split"]["method"]

    if cfg["dataset"]["split"]["train_name"] and cfg["dataset"]["split"]["valid_name"] and cfg["dataset"]["split"]["test_name"]:
        dataset_train = eval(cfg["dataset"]["tier2_task"])(name = cfg["dataset"]["split"]["train_name"], \
                            path = cfg["dataset"]["path"]).get_data()
        dataset_valid = eval(cfg["dataset"]["tier2_task"])(name = cfg["dataset"]["split"]["valid_name"], \
                            path = cfg["dataset"]["path"]).get_data()
        dataset_test = eval(cfg["dataset"]["tier2_task"])(name = cfg["dataset"]["split"]["test_name"], \
                            path = cfg["dataset"]["path"]).get_data()
        dataset_dic = {"train": dataset_train, "valid": dataset_valid, "test": dataset_test}
    else:
        dataset_dic = dataset.get_split(seed=cfg["seed"], label_type=label_type, \
            num_class=num_class, **cfg["dataset"]["split"])

    return dataset_dic, dataset
    # train_set = Dataset(cfg=cfg, data_df=dataset_dic["train"], entities=dataset.entities, \
    #     entity_type=dataset.entity_type, mode="train")
    # valid_set = Dataset(cfg=cfg, data_df=dataset_dic["valid"], entities=dataset.entities, \
    #     entity_type=dataset.entity_type, mode="valid")
    # test_set = Dataset(cfg=cfg, data_df=dataset_dic["test"], entities=dataset.entities, \
    #     entity_type=dataset.entity_type, mode="test")

    # return {"train_set": train_set, "valid_set": valid_set, "test_set": test_set}

def get_category_list(dataset: torch.utils.data.Dataset):
    label_name = dataset.label_name
    class_label = np.array(dataset.data[label_name])
    num_class = len(set(class_label))
    num_list = []
    for i in range(num_class):
        num_list.append(np.sum(class_label == i))
    return num_list

def set_distributed(cfg, local_rank, model_dir, tensorboard_dir):
    """for distributed training
    """
    if local_rank == 0:

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )

            if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)
        print("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )


    if cfg["train"]["distributed"]:
        if local_rank == 0:
            print("Init the process group for distributed training")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        if local_rank == 0:
            print("Init complete")

def set_resume(cfg, optimizer, scheduler, start_epoch, best_result, best_epoch, model_dir, auto_resume):
    """for resumed training
    """

    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, "epoch_{}.pth".format(resume_epoch))

    if cfg["resume_model"] != "" or auto_resume:
        if cfg["resume_model"] == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg["resume_model"] if "/" in cfg["resume_model"] else \
                os.path.join(model_dir, cfg["resume_model"])
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(
            resume_model, map_location="cpu" if cfg["train"]["distributed"] else "cuda"
        )
        model.module.load_model(resume_model)
        if cfg["resume_mode"] != "state_dict":
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if use_apex: amp.load_state_dict(checkpoint["amp"])
            start_epoch = checkpoint["epoch"] + 1
            best_result = checkpoint["best_result"]
            best_epoch = checkpoint["best_epoch"]
    return optimizer, scheduler, start_epoch, best_result, best_epoch

def generate_config(drug_encoding = None, target_encoding = None, 
                    result_folder = "./result/",
                    input_dim_drug = 1024, 
                    input_dim_protein = 8420,
                    hidden_dim_drug = 256, 
                    hidden_dim_protein = 256,
                    max_len_drug = 50,
                    max_len_protein = 545,
                    cls_hidden_dims = [1024, 1024, 512],
                    mlp_hidden_dims_drug = [1024, 256, 64],
                    mlp_hidden_dims_target = [1024, 256, 64],
                    batch_size = 256,
                    train_epoch = 10,
                    test_every_X_epoch = 20,
                    LR = 1e-4,
                    decay = 0,
                    transformer_emb_size_drug = 128,
                    transformer_intermediate_size_drug = 512,
                    transformer_num_attention_heads_drug = 8,
                    transformer_n_layer_drug = 8,
                    transformer_emb_size_target = 64,
                    transformer_intermediate_size_target = 256,
                    transformer_num_attention_heads_target = 4,
                    transformer_n_layer_target = 2,
                    transformer_dropout_rate = 0.1,
                    transformer_attention_probs_dropout = 0.1,
                    transformer_hidden_dropout_rate = 0.1,
                    mpnn_hidden_size = 50,
                    mpnn_depth = 3,
                    cnn_drug_filters = [32,64,96],
                    cnn_drug_kernels = [4,6,8],
                    cnn_target_filters = [32,64,96],
                    cnn_target_kernels = [4,8,12],
                    rnn_Use_GRU_LSTM_drug = "GRU",
                    rnn_drug_hid_dim = 64,
                    rnn_drug_n_layers = 2,
                    rnn_drug_bidirectional = True,
                    rnn_Use_GRU_LSTM_target = "GRU",
                    rnn_target_hid_dim = 64,
                    rnn_target_n_layers = 2,
                    rnn_target_bidirectional = True,
                    num_workers = 0,
                    cuda_id = None,
                    gnn_hid_dim_drug = 64,
                    gnn_num_layers = 3,
                    gnn_activation = F.relu,
                    neuralfp_max_degree = 10,
                    neuralfp_predictor_hid_dim = 128,
                    neuralfp_predictor_activation = torch.tanh,
                    attentivefp_num_timesteps = 2,
                    **kwargs
                    ):

    base_config = {"input_dim_drug": input_dim_drug,
                    "input_dim_protein": input_dim_protein,
                    "hidden_dim_drug": hidden_dim_drug, # hidden dim of drug
                    "hidden_dim_protein": hidden_dim_protein, # hidden dim of protein
                    "max_len_drug": max_len_drug,
                    "max_len_protein": max_len_protein,
                    "cls_hidden_dims" : cls_hidden_dims, # decoder classifier dim 1
                    "batch_size": batch_size,
                    "train_epoch": train_epoch,
                    "test_every_X_epoch": test_every_X_epoch, 
                    "LR": LR,
                    "drug_encoding": drug_encoding,
                    "target_encoding": target_encoding, 
                    "result_folder": result_folder,
                    "binary": False,
                    "num_workers": num_workers,
                    "cuda_id": cuda_id  
    }
    if not os.path.exists(base_config["result_folder"]):
        os.makedirs(base_config["result_folder"])
    
    if drug_encoding == "Morgan":
        base_config["mlp_hidden_dims_drug"] = mlp_hidden_dims_drug # MLP classifier dim 1				
    elif drug_encoding == "ErG":
        base_config["input_dim_drug"] = 315
        base_config["mlp_hidden_dims_drug"] = mlp_hidden_dims_drug # MLP classifier dim 1
    elif drug_encoding == "Pubchem":
        base_config["input_dim_drug"] = 881
        base_config["mlp_hidden_dims_drug"] = mlp_hidden_dims_drug # MLP classifier dim 1				
    elif drug_encoding == "Daylight":
        base_config["input_dim_drug"] = 2048
        base_config["mlp_hidden_dims_drug"] = mlp_hidden_dims_drug # MLP classifier dim 1						
    elif drug_encoding == "rdkit_2d_normalized":
        base_config["input_dim_drug"] = 200
        base_config["mlp_hidden_dims_drug"] = mlp_hidden_dims_drug # MLP classifier dim 1
    elif drug_encoding == "ESPF":
        base_config["input_dim_drug"] = len(idx2word_d)
        base_config["mlp_hidden_dims_drug"] = mlp_hidden_dims_drug # MLP classifier dim 1				
    elif drug_encoding == "CNN":
        base_config["cnn_drug_filters"] = cnn_drug_filters
        base_config["cnn_drug_kernels"] = cnn_drug_kernels
    elif drug_encoding == "CNN_RNN":
        base_config["rnn_Use_GRU_LSTM_drug"] = rnn_Use_GRU_LSTM_drug
        base_config["rnn_drug_hid_dim"] = rnn_drug_hid_dim
        base_config["rnn_drug_n_layers"] = rnn_drug_n_layers
        base_config["rnn_drug_bidirectional"] = rnn_drug_bidirectional 
        base_config["cnn_drug_filters"] = cnn_drug_filters
        base_config["cnn_drug_kernels"] = cnn_drug_kernels
    elif drug_encoding == "Transformer":
        base_config["input_dim_drug"] = 2586
        base_config["transformer_emb_size_drug"] = transformer_emb_size_drug
        base_config["transformer_num_attention_heads_drug"] = transformer_num_attention_heads_drug
        base_config["transformer_intermediate_size_drug"] = transformer_intermediate_size_drug
        base_config["transformer_n_layer_drug"] = transformer_n_layer_drug
        base_config["transformer_dropout_rate"] = transformer_dropout_rate
        base_config["transformer_attention_probs_dropout"] = transformer_attention_probs_dropout
        base_config["transformer_hidden_dropout_rate"] = transformer_hidden_dropout_rate
        base_config["hidden_dim_drug"] = transformer_emb_size_drug
    elif drug_encoding == "MPNN":
        base_config["hidden_dim_drug"] = hidden_dim_drug
        base_config["batch_size"] = batch_size 
        base_config["mpnn_hidden_size"] = mpnn_hidden_size
        base_config["mpnn_depth"] = mpnn_depth
        #raise NotImplementedError
    elif drug_encoding == "DGL_GCN":
        base_config["gnn_hid_dim_drug"] = gnn_hid_dim_drug
        base_config["gnn_num_layers"] = gnn_num_layers
        base_config["gnn_activation"] = gnn_activation
    elif drug_encoding == "DGL_NeuralFP":
        base_config["gnn_hid_dim_drug"] = gnn_hid_dim_drug
        base_config["neuralfp_max_degree"] = neuralfp_max_degree
        base_config["gnn_activation"] = gnn_activation
        base_config["neuralfp_predictor_hid_dim"] = neuralfp_predictor_hid_dim
        base_config["gnn_num_layers"] = gnn_num_layers
        base_config["neuralfp_predictor_activation"] = neuralfp_predictor_activation
    elif drug_encoding == "DGL_GIN_AttrMasking":
        ## loaded pretrained model specifications
        pass
    elif drug_encoding == "DGL_GIN_ContextPred":
        ## loaded pretrained model specifications
        pass
    elif drug_encoding == "DGL_AttentiveFP":
        base_config["gnn_hid_dim_drug"] = gnn_hid_dim_drug
        base_config["gnn_num_layers"] = gnn_num_layers
        base_config["attentivefp_num_timesteps"] = attentivefp_num_timesteps
    elif drug_encoding is None:
        pass
    else:
        raise AttributeError("Please use the correct drug encoding available!")

    if target_encoding == "AAC":
        base_config["mlp_hidden_dims_target"] = mlp_hidden_dims_target # MLP classifier dim 1				
    elif target_encoding == "PseudoAAC":
        base_config["input_dim_protein"] = 30
        base_config["mlp_hidden_dims_target"] = mlp_hidden_dims_target # MLP classifier dim 1				
    elif target_encoding == "Conjoint_triad":
        base_config["input_dim_protein"] = 343
        base_config["mlp_hidden_dims_target"] = mlp_hidden_dims_target # MLP classifier dim 1				
    elif target_encoding == "Quasi-seq":
        base_config["input_dim_protein"] = 100
        base_config["mlp_hidden_dims_target"] = mlp_hidden_dims_target # MLP classifier dim 1	
    elif target_encoding == "ESPF":
        base_config["input_dim_protein"] = len(idx2word_p)
        base_config["mlp_hidden_dims_target"] = mlp_hidden_dims_target # MLP classifier dim 1			
    elif target_encoding == "CNN":
        base_config["cnn_target_filters"] = cnn_target_filters
        base_config["cnn_target_kernels"] = cnn_target_kernels
    elif target_encoding == "CNN_RNN":
        base_config["rnn_Use_GRU_LSTM_target"] = rnn_Use_GRU_LSTM_target
        base_config["rnn_target_hid_dim"] = rnn_target_hid_dim
        base_config["rnn_target_n_layers"] = rnn_target_n_layers
        base_config["rnn_target_bidirectional"] = rnn_target_bidirectional 
        base_config["cnn_target_filters"] = cnn_target_filters
        base_config["cnn_target_kernels"] = cnn_target_kernels
    elif target_encoding == "MLP":
        base_config["input_dim_protein"] = max_len_protein # default value in DeepPurpose
        base_config["hidden_dim_protein"] = mlp_hidden_dims_target[-1] # to be consistent with the head
        base_config["mlp_hidden_dims_target"] = mlp_hidden_dims_target[:-1] # MLP classifier dim 1
    elif target_encoding == "Transformer":
        base_config["input_dim_protein"] = 4114
        base_config["transformer_emb_size_target"] = transformer_emb_size_target
        base_config["transformer_num_attention_heads_target"] = transformer_num_attention_heads_target
        base_config["transformer_intermediate_size_target"] = transformer_intermediate_size_target
        base_config["transformer_n_layer_target"] = transformer_n_layer_target	
        base_config["transformer_dropout_rate"] = transformer_dropout_rate
        base_config["transformer_attention_probs_dropout"] = transformer_attention_probs_dropout
        base_config["transformer_hidden_dropout_rate"] = transformer_hidden_dropout_rate
        base_config["hidden_dim_protein"] = transformer_emb_size_target
    elif target_encoding == "ESM" or "ESM_embed":
        model = EsmModel.from_pretrained(kwargs["ESM_model"])
        input_dim_protein = model.__dict__["config"].__dict__["hidden_size"]
        base_config["ESM_model"] = kwargs["ESM_model"]
        base_config["mlp_hidden_dims_target"] = mlp_hidden_dims_target
        base_config["input_dim_protein"] = input_dim_protein
        if target_encoding == "ESM":
            base_config["hidden_dim_protein"] = input_dim_protein # to be consistent with the head
    elif target_encoding is None:
        pass
    else:
        raise AttributeError("Please use the correct protein encoding available!")
    return base_config

def set_baseline(exp_params, cfg):
    if exp_params["baseline"] == "Default_cls":
        cfg["loss"]["type"] = "CrossEntropy"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "Default_reg":
        cfg["loss"]["type"] = "MSE"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "BalancedSoftmax":
        cfg["loss"]["type"] = "BalancedSoftmaxCE"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "ClassBalanced":
        cfg["loss"]["type"] = "ClassBalancedFocal"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "CostSensitive":
        cfg["loss"]["type"] = "CostSensitiveCE"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "InfluenceBalanced":
        cfg["loss"]["type"] = "InfluenceBalancedLoss"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = True
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "Mixup_cls":
        cfg["loss"]["type"] = "CrossEntropy"
        cfg["train"]["combiner"]["type"] = "mix_up"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "Mixup_reg":
        cfg["loss"]["type"] = "MSE"
        cfg["train"]["combiner"]["type"] = "mix_up"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "Remix_cls":
        cfg["loss"]["type"] = "CrossEntropy"
        cfg["train"]["combiner"]["type"] = "remix"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "Remix_reg":
        cfg["loss"]["type"] = "MSE"
        cfg["train"]["combiner"]["type"] = "remix"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "BBN_cls":
        cfg["loss"]["type"] = "CrossEntropy"
        cfg["train"]["combiner"]["type"] = "bbn_mix"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "BBN_reg":
        cfg["loss"]["type"] = "MSE"
        cfg["train"]["combiner"]["type"] = "bbn_mix"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "CDT":
        cfg["loss"]["type"] = "CDT"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = True
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "Decoupling":
        cfg["loss"]["type"] = "CrossEntropy"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = True
        cfg["train"]["sampler"]["type"] = "weighted_sampler"
        cfg["train"]["sampler"]["weighted_sampler"]["type"] = "progressive"

    elif exp_params["baseline"] == "DiVE":
        cfg["loss"]["type"] = "DiVEKLD"
        cfg["train"]["combiner"]["type"] = "dive"
        cfg["train"]["combiner"]["dive"]["teacher_model"] = ""
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "Focal-R":
        cfg["loss"]["type"] = "FocalR"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "LDS":
        cfg["loss"]["type"] = "LDS"
        cfg["train"]["combiner"]["type"] = "default"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    elif exp_params["baseline"] == "FDS":
        cfg["loss"]["type"] = "MSE"
        cfg["train"]["combiner"]["type"] = "FDS"
        cfg["train"]["two_stage"]["drw"] = False
        cfg["train"]["two_stage"]["drs"] = False

    return cfg


######################
# Stolen from Deeppurpose.utils
######################

# '?' padding
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

from sklearn.preprocessing import OneHotEncoder
enc_protein = OneHotEncoder(dtype=np.int16).fit(np.array(amino_char).reshape(-1, 1)) # for memory-saving
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def protein_2_embed(x):
	return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T    
