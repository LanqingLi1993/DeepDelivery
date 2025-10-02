import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.colors import LinearSegmentedColormap  
import os
import torch
from misc import domain_map
from lrp import lrp_score, att_score


def generate_lrp_score(
    i_layer = 0,
    num_class = 2,
    target_class = 1,
    normalize = 0,
    softmax = 1,
    output_dir = './output/uniprotkb_trim_AND_reviewed_true_2024_12_04/',
    offset_lst = [0, 3, 3, 10, 10, 21, 21], # based on conv layer kernel size and stride,
    targetclassonly = 0,
):

    import _init_paths
    from DeepPurpose.utils import trans_protein, protein_2_embed
    from utils.featurizer import protein2emb_encoder, protein2emb_encoder_custom
    from utils.utils import (
    deep_update_dict,
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
    get_dataset,
    set_baseline
    )
    import argparse
    from config import config, SimpleConfig
    import json, os
    import os.path as osp
    import pandas as pd
    from misc import get_segment, get_seq

    path = "./"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=path)
    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)
    cfg = set_baseline(exp_params, cfg)
    target_encoding = cfg['dataset']['protein_encoding']


    if target_encoding == "CNN":
        # df = pd.read_pickle("./data/uniprotkb_trim_AND_reviewed_true_2024_12_04_processed_full-v4.pkl") # trim data without esm embedding
        # key_lst = ['Entry',	'Entry Name', 'X', 'Subclass', 'Predicted Score', 'lrp_score', 'lrp_segments', 'lrp_aa_segments']
        # origin_key_lst = ['Entry',	'Entry Name', 'X', 'Subclass']
        df = pd.read_pickle("./xai/data/uniprotkb_trim_AND_reviewed_true_2024_12_04_processed_full-v5.pkl") # trim data without esm embedding
        key_lst = ['ID',	'Entry Name', 'X', 'Subclass', 'Predicted Score', 'lrp_score', 'lrp_segments', 'lrp_aa_segments']
        origin_key_lst = ['ID',	'Entry Name', 'X', 'Subclass']
    elif target_encoding == "ESM_embed":
        df = pd.read_pickle("./data/uniprotkb_trim_AND_reviewed_true_2024_12_04_processed_full-v4-ESM_embed.pkl")  # trim data with esm embedding
        key_lst = ['Entry',	'Entry Name', 'X', 'X_seq', 'Subclass', 'Predicted Score', 'lrp_score', 'lrp_segments', 'lrp_aa_segments']
        origin_key_lst = ['Entry',	'Entry Name', 'X_seq', 'Subclass']
    elif target_encoding == "Transformer":
        df = pd.read_pickle("./xai/data/uniprotkb_trim_AND_reviewed_true_2024_12_04_processed_full-v5.pkl") # trim data without esm embedding
        key_lst = ['ID',	'Entry Name', 'X', 'Subclass', 'Predicted Score', 'lrp_score', 'lrp_segments', 'lrp_aa_segments', 'att_score', 'att_segments', 'att_aa_segments']
        origin_key_lst = ['ID',	'Entry Name', 'X', 'Subclass']
    else:
        NotImplementedError
    # df['Predicted Score'] = np.nan


    entity_type = ['protein']
    local_rank = cfg["train"]["local_rank"]
    rank = local_rank
    logger, log_file, exp_id = create_logger(cfg, local_rank, test=True)
    cfg["setting"]["num_class"] = 2

    model_dir = osp.join(cfg["output_dir"], cfg["dataset"]["dataset_name"], "models", cfg["test"]["exp_id"])

    model = get_model(cfg=cfg, device=torch.device('cpu'), logger=logger, entity_type=entity_type)
    model.eval()


    # Load self-trained model
    if not cfg["network"]["pretrained"]:
        model_file = os.path.join(model_dir, cfg["test"]["model_file"])
        model.load_model(model_file, map_location="cpu")
        # model = torch.nn.DataParallel(model).cuda()

    max_seq_len = cfg["backbone"]["deeppurpose"]["max_len_protein"]
    target_encoding = cfg['dataset']['protein_encoding']
    save_dir = os.path.join(output_dir, "test", cfg["test"]["exp_id"])

    os.makedirs(save_dir, exist_ok=True)

    
    if target_encoding == "Transformer":
        att_score(model=model, i_layer=i_layer, target_class=target_class, df=df, 
                  key_lst=key_lst, origin_key_lst=origin_key_lst, target_encoding=target_encoding,
                  output_dir=save_dir,
                  normalize=normalize,
                  bpe_encoding=0,
                  targetclassonly=targetclassonly)
    else:
        lrp_score(model=model, i_layer=i_layer, num_class=num_class, target_class=target_class, df=df, 
            key_lst=key_lst, origin_key_lst=origin_key_lst, target_encoding=target_encoding,
            output_dir=save_dir,
            normalize=normalize,
            bpe_encoding=0,
            offset_lst=offset_lst,
            softmax=softmax,
            targetclassonly=targetclassonly)
    

def plot_lrp_heatmap(
        dataset_name = "uniprotkb_trim_AND_reviewed_true_2024_12_04",
        model_name = "PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup_minus_uniprot_9606_2023_10_12_DGL_GCN_CrossEntropy_2_MLP_2025-04-13-10-23-04-239741",
        file_name = "lrp_scores-layer0-targetclass1-normalize0-softmax1-v6",
        threshold = 0.9,
        na_value = None):

    file_path = "./output/{}/test/{}/{}.pkl".format(dataset_name, model_name, file_name)

    # 打开并读取 .pkl 文件
    df = pd.read_pickle(file_path)
    df.to_csv("./output/uniprotkb_trim_AND_reviewed_true_2024_12_04/test/{}/{}.csv".format(model_name, file_name))

    df = df.dropna() # 去掉包含空值的行（默认最后一行）

    # 将 'Entry Name' 设置为行索引
    df.set_index('Entry Name', inplace=True)

    def get_predict_score(score_lst):
        if not isinstance(score_lst, np.ndarray):
            print(score_lst)
        return score_lst[1]

    def get_len(x):
        return len(x)
    
    df['capsid score'] = df['Predicted Score'].apply(get_predict_score)
    df = df[df['capsid score'] > threshold]
    df['len'] = df['X'].apply(get_len)
    # df = df[(df['len'] >= 100) * (df['len'] <= 1024)]
    if file_name.startswith('att'):
        df.drop(columns=['ID', 'X', 'Subclass', 'Predicted Score', 'att_score','att_segments', 'att_aa_segments'], inplace=True)
    elif file_name.startswith('lrp'):
        df.drop(columns=['ID', 'X', 'Subclass', 'Predicted Score', 'lrp_score','lrp_segments', 'lrp_aa_segments'], inplace=True)
    else:
        NotImplementedError
    df.drop(columns=['len', 'capsid score'], inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  
        print('domain: {}, freq: {}'.format(col, (df[col] != 0).sum())) # 统计每个domain的频率

    if na_value is not None:
        df = df.replace(0, na_value) # 将缺省值替换为na_value

    # # 绘制热图
    plt.figure(figsize=(10, 8))

    colors = ["blue", "red"]  
    blue_red_cmap = LinearSegmentedColormap.from_list("blue_red", colors) 
    # sns.heatmap(df[:-1], cmap=sns.color_palette("light:b", as_cmap=True), robust=True)
    sns.heatmap(df, cmap=blue_red_cmap, robust=True)
                # vmin=-0.00000002,  vmax=0.00000002)  # `annot=True` 显示数值，`fmt=".2f"` 格式化为小数点两位


    # 旋转列标签 90 度
    plt.xticks(rotation=90)

    # 添加标题
    plt.title("Heatmap of CSV Data")

    # 显示图表
    plt.tight_layout()  # 调整布局，避免标签重叠
    # plt.show()
    save_dir = os.path.join("./xai/output", model_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "{}.pdf".format(file_name)))

if __name__ == "__main__":
    # run the following blocks in sequence

    ###########
    # generate domain mapping
    ###########

    # df_path = "./data/uniprotkb_trim_AND_reviewed_true_2024_12_04.csv"
    # subclass_df_path = "./xai/data/Labeled-uniprotkb_TRIM.xlsx"
    # map_df_path = "./xai/data/domain_mapping-v3.xlsx"
    # save_dir = "./xai/data/uniprotkb_trim_AND_reviewed_true_2024_12_04_processed_full-v5.pkl"
    # domain_map(df_path, subclass_df_path, map_df_path, save_dir)

    ###########
    # generate lrp score
    ###########
    # from project directory run 'python xai/lrp_vis.py --config ./lib/config/CNN-lrp-llq.json'
    i_layer = 0
    num_class = 2
    target_class = 1
    normalize = 0
    softmax = 0
    offset_lst = [0, 3, 3, 10, 10, 21, 21] # based on conv layer kernel size and stride
    output_dir = "./output/uniprotkb_trim_AND_reviewed_true_2024_12_04"
    targetclassonly = 1
    generate_lrp_score(
        i_layer = i_layer,
        num_class = num_class,
        target_class = target_class,
        normalize = normalize,
        softmax = softmax,
        offset_lst = offset_lst,
        output_dir = output_dir,
        targetclassonly = targetclassonly
    )


    # ##########
    # # plot lrp heatmap
    # ##########

    # dataset_name = "uniprotkb_trim_AND_reviewed_true_2024_12_04"
    # # model_name = "PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup_minus_uniprot_9606_2023_10_12_DGL_GCN_CrossEntropy_2_MLP_2025-04-13-10-23-04-239741"
    # # model_name = "PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k_DGL_GCN_CrossEntropy_2_MLP_2025-01-14-17-38-34-027507"
    # # model_name = "PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup_minus_uniprot_9606_2023_10_12_DGL_GCN_CrossEntropy_2_MLP_2025-04-15-14-34-50-769591"
    # model_name = "PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup_minus_uniprot_9606_2023_10_12_DGL_GCN_CrossEntropy_2_MLP_2025-04-13-11-31-47-526709"
    # file_name = "att_scores-layer0-targetclass1-normalize0-targetclassonly1-v6"
    # threshold = 0
    # na_value = -0.0001
    # plot_lrp_heatmap(
    #     dataset_name=dataset_name,
    #     model_name=model_name,
    #     file_name=file_name,
    #     threshold=threshold,
    #     na_value=na_value
    #     )

