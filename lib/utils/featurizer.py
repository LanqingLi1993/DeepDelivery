import pandas as pd
from DeepPurpose.utils import *
import re

amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

def encode_protein(df_data, target_encoding, cfg, column_name = "Target Sequence", save_column_name = "target_encoding"):
	print("encoding protein...")
	print(df_data)
	print("unique target sequence: " + str(len(df_data[column_name].unique())))
	if target_encoding == "AAC":
		print("-- Encoding AAC takes time. Time Reference: 24s for ~100 sequences in a CPU.\
				 Calculate your time by the unique target sequence #, instead of the entire dataset.")
		AA = pd.Series(df_data[column_name].unique()).apply(target2aac)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == "PseudoAAC":
		print("-- Encoding PseudoAAC takes time. Time Reference: 462s for ~100 sequences in a CPU.\
				 Calculate your time by the unique target sequence #, instead of the entire dataset.")
		AA = pd.Series(df_data[column_name].unique()).apply(target2paac)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == "Conjoint_triad":
		AA = pd.Series(df_data[column_name].unique()).apply(target2ct)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == "Quasi-seq":
		AA = pd.Series(df_data[column_name].unique()).apply(target2quasi)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == "ESPF":
		AA = pd.Series(df_data[column_name].unique()).apply(protein2espf)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == "CNN":
		AA = pd.Series(df_data[column_name].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 
	elif target_encoding == "CNN_RNN":
		AA = pd.Series(df_data[column_name].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == "MLP":
		AA = pd.Series(df_data[column_name].unique()).apply(lambda x: protein2emb_encoder_float32(x, cfg["max_len_protein"]))
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]		
	elif target_encoding == "Transformer":
		AA = pd.Series(df_data[column_name].unique()).apply(lambda x: protein2emb_encoder(x, cfg["max_len_protein"])) # bpe encoding
		# AA = pd.Series(df_data[column_name].unique()).apply(lambda x: protein2emb_encoder_custom(x, cfg["max_len_protein"])) # aa-wise encoding
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == "ESM":
		from transformers import AutoTokenizer, EsmModel
		tokenizer = AutoTokenizer.from_pretrained(cfg["ESM_model"])
		# model = EsmModel.from_pretrained(cfg["ESM_model"])
		model_max_length = tokenizer.__dict__["model_max_length"]
		max_length = min(model_max_length, cfg["max_len_protein"])
		print("max_length", max_length)
		# If token indices sequence length is longer than the specified maximum sequence length for this model, running this sequence through the model will result in indexing errors
		AA = pd.Series(df_data[column_name].unique()).apply(lambda x: (tokenizer(x, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)["input_ids"][0].numpy().astype(np.int_),
															tokenizer(x, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)["attention_mask"][0].numpy().astype(np.int_)))
		# AA_emb = AA.apply(lambda x: list(torch.mean(model(**x).last_hidden_state[0], dim=0))) # mean pooling of the last hidden state (shape: (batch_size, seq_len, emb_dim))
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	elif target_encoding == "ESM_embed":
		def str2arr(embed_str):
			p = re.compile('[\[\]\n]')
			embed = p.sub(r'', embed_str)
			return np.array([float(n) for n in embed.split()])
		AA = pd.Series(df_data[column_name].unique()).apply(str2arr)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
	else:
		raise AttributeError("Please use the correct protein encoding available!")
	return df_data

def protein2emb_encoder(x, max_p=545):
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), "constant", constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)


def protein2emb_encoder_custom(x, max_p=545):
    '''
	Custom protein2emb_encoder function for protein tokenization
    '''
    t1 = list(x)
    words2idx_p = dict(zip(amino_char, range(0, len(amino_char))))
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), "constant", constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def protein2emb_encoder_float32(x, max_p=545):
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), "constant", constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i.astype(np.float32)

