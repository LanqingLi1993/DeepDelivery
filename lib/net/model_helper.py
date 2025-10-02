import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
import collections
import os

from DeepPurpose.encoders import *
from head.classifier import LWS, FCNorm, MLP, Identity

torch.manual_seed(1)
np.random.seed(1)

def model_initialize(**config):
    model = Protein_Prediction(**config)
    return model

##########################################################
# Customized Models, adapted from DeepPurpose.model_helper 
##########################################################

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # print("att_prob", attention_probs, attention_probs.shape)
        self.att_scores = attention_probs

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
    

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        # print("att", attention_output, attention_output.shape)
        intermediate_output = self.intermediate(attention_output)
        # print("itmd", intermediate_output, intermediate_output.shape)
        layer_output = self.output(intermediate_output, attention_output)
        # print("layer", layer_output, layer_output.shape)
        return layer_output    

    
class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        att_scores = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            att_scores.append(layer_module.attention.self.att_scores)
            # print("att_scores", att_scores[-1], att_scores[-1].shape)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states, att_scores

class transformer(nn.Sequential):
    def __init__(self, encoding, **config):
        super(transformer, self).__init__()
        if encoding == "drug":
            # self.emb = Embeddings(config["input_dim_drug"], config["transformer_emb_size_drug"], 50, config["transformer_dropout_rate"])
            self.emb = Embeddings(config["input_dim_drug"], config["transformer_emb_size_drug"], 
                                   config["max_len_drug"], config["transformer_dropout_rate"])
            self.encoder = Encoder_MultipleLayers(config["transformer_n_layer_drug"], 
                                                    config["transformer_emb_size_drug"], 
                                                    config["transformer_intermediate_size_drug"], 
                                                    config["transformer_num_attention_heads_drug"],
                                                    config["transformer_attention_probs_dropout"],
                                                    config["transformer_hidden_dropout_rate"])
        elif encoding == "protein":
            self.emb = Embeddings(config["input_dim_protein"], config["transformer_emb_size_target"], 
                                   config["max_len_protein"], config["transformer_dropout_rate"])
            self.encoder = Encoder_MultipleLayers(config["transformer_n_layer_target"], 
                                                    config["transformer_emb_size_target"], 
                                                    config["transformer_intermediate_size_target"], 
                                                    config["transformer_num_attention_heads_target"],
                                                    config["transformer_attention_probs_dropout"],
                                                    config["transformer_hidden_dropout_rate"])

    ### parameter v (tuple of length 2) is from utils.drug2emb_encoder 
    def forward(self, v, pooling=1):
        e = v[:, 0].long()
        e_mask =  v[:, 1].long()
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0
        emb = self.emb(e)
        encoded_layers, att_mat = self.encoder(emb.float(), ex_e_mask.float())
        # return encoded_layers[:,0], att_mat
        # print('encoded_layers', encoded_layers.shape)
        if pooling:
            return F.adaptive_avg_pool1d(encoded_layers.transpose(1, 2), output_size=1), att_mat # n_batch * hidden_dim
            # return encoded_layers[:,0], att_mat
        else:
            return encoded_layers.transpose(1, 2), att_mat # n_batch * hidden_dim * seq_len


class Protein_Prediction:
    """
        Protein Function Prediction 
    """

    def __init__(self, **config):
        target_encoding = config["target_encoding"]

        if target_encoding == "AAC" or target_encoding == "PseudoAAC" or  target_encoding == "Conjoint_triad" or target_encoding == "Quasi-seq" or target_encoding == "ESPF" or target_encoding == "MLP":
            self.model_protein = MLP(config["input_dim_protein"], config["hidden_dim_protein"], config["mlp_hidden_dims_target"])
        elif target_encoding == "CNN":
            # self.model_protein = CNN("protein", **config)
            from backbone import CNN
            self.model_protein = CNN("protein", **config)
        elif target_encoding == "CNN_RNN":
            self.model_protein = CNN_RNN("protein", **config)
        elif target_encoding == "Transformer":
            self.model_protein = transformer("protein", **config)
        elif target_encoding == "ESM":
            from transformers import EsmModel
            model = EsmModel.from_pretrained(config["ESM_model"])
            # input_dim_protein = model.__dict__["config"].__dict__["hidden_size"]
            # self.model_protein = MLP(input_dim_protein, config["hidden_dim_protein"], config["mlp_hidden_dims_target"])
            self.model_protein = model
        elif target_encoding == "ESM_embed":
            self.model_protein = MLP(config["input_dim_protein"], config["hidden_dim_protein"], config["mlp_hidden_dims_target"])
        else:
            raise AttributeError("Please use one of the available encoding method.")

        # self.model = Classifier(self.model_protein, **config)
        self.config = config
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.target_encoding = target_encoding
        self.result_folder = config["result_folder"]
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)            
        self.binary = False
        if "num_workers" not in self.config.keys():
            self.config["num_workers"] = 0
        if "decay" not in self.config.keys():
            self.config["decay"] = 0

    # def test_(self, data_generator, model, repurposing_mode = False, test = False, verbose = True):
    # 	y_pred = []
    # 	y_label = []
    # 	model.eval()
    # 	for i, (v_p, label) in enumerate(data_generator):
    # 		if self.target_encoding == "Transformer":
    # 			v_p = v_p
    # 		else:
    # 			v_p = v_p.float().to(self.device)              
    # 		score = self.model(v_p)

    # 		if self.binary:
    # 			m = torch.nn.Sigmoid()
    # 			logits = torch.squeeze(m(score)).detach().cpu().numpy()
    # 		else:
    # 			logits = torch.squeeze(score).detach().cpu().numpy()

    # 		label_ids = label.to("cpu").numpy()
    # 		y_label = y_label + label_ids.flatten().tolist()
    # 		y_pred = y_pred + logits.flatten().tolist()
    # 		outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        
    # 	model.train()
    # 	if self.binary:
    # 		if repurposing_mode:
    # 			return y_pred
    # 		## ROC-AUC curve
    # 		if test:
    # 			if verbose:
    # 				roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
    # 				plt.figure(0)
    # 				roc_curve(y_pred, y_label, roc_auc_file, self.target_encoding)
    # 				plt.figure(1)
    # 				pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
    # 				prauc_curve(y_pred, y_label, pr_auc_file, self.target_encoding)

    # 		return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred
    # 	else:
    # 		if repurposing_mode:
    # 			return y_pred
    # 		return mean_squared_error(y_label, y_pred), \
    # 			   pearsonr(y_label, y_pred)[0], \
    # 			   pearsonr(y_label, y_pred)[1], \
    # 			   concordance_index(y_label, y_pred), y_pred

    # def train(self, train, val, test = None, verbose = True):
    # 	if len(train.Label.unique()) == 2:
    # 		self.binary = True
    # 		self.config["binary"] = True

    # 	lr = self.config["LR"]
    # 	decay = self.config["decay"]

    # 	BATCH_SIZE = self.config["batch_size"]
    # 	train_epoch = self.config["train_epoch"]
    # 	if "test_every_X_epoch" in self.config.keys():
    # 		test_every_X_epoch = self.config["test_every_X_epoch"]
    # 	else:     
    # 		test_every_X_epoch = 40
    # 	loss_history = []

    # 	self.model = self.model.to(self.device)

    # 	# support multiple GPUs
    # 	if torch.cuda.device_count() > 1:
    # 		if verbose:
    # 			print("Let"s use " + str(torch.cuda.device_count()) + " GPUs!")
    # 		self.model = nn.DataParallel(self.model, dim = 0)
    # 	elif torch.cuda.device_count() == 1:
    # 		if verbose:
    # 			print("Let"s use " + str(torch.cuda.device_count()) + " GPU!")
    # 	else:
    # 		if verbose:
    # 			print("Let"s use CPU/s!")
    # 	# Future TODO: support multiple optimizers with parameters
    # 	opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)

    # 	if verbose:
    # 		print("--- Data Preparation ---")

    # 	params = {"batch_size": BATCH_SIZE,
    #     		"shuffle": True,
    #     		"num_workers": self.config["num_workers"],
    #     		"drop_last": False}
        
    # 	training_generator = data.DataLoader(data_process_loader_Protein_Prediction(train.index.values, 
    # 																				 train.Label.values, 
    # 																				 train, **self.config), 
    # 																					**params)
    # 	validation_generator = data.DataLoader(data_process_loader_Protein_Prediction(val.index.values, 
    # 																					val.Label.values, 
    # 																					val, **self.config), 
    # 																					**params)
        
    # 	if test is not None:
    # 		info = data_process_loader_Protein_Prediction(test.index.values, test.Label.values, test, **self.config)
    # 		params_test = {"batch_size": BATCH_SIZE,
    # 				"shuffle": False,
    # 				"num_workers": self.config["num_workers"],
    # 				"drop_last": False,
    # 				"sampler":SequentialSampler(info)}
    # 		testing_generator = data.DataLoader(data_process_loader_Protein_Prediction(test.index.values, test.Label.values, test, **self.config), **params_test)

    # 	# early stopping
    # 	if self.binary:
    # 		max_auc = 0
    # 	else:
    # 		max_MSE = 10000
    # 	model_max = copy.deepcopy(self.model)

    # 	valid_metric_record = []
    # 	valid_metric_header = ["# epoch"] 
    # 	if self.binary:
    # 		valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
    # 	else:
    # 		valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
    # 	table = PrettyTable(valid_metric_header)
    # 	float2str = lambda x:"%0.4f"%x

    # 	if verbose:
    # 		print("--- Go for Training ---")
    # 	t_start = time() 
    # 	for epo in range(train_epoch):
    # 		for i, (v_p, label) in enumerate(training_generator):
                
    # 			if self.target_encoding == "Transformer":
    # 				v_p = v_p
    # 			else:
    # 				v_p = v_p.float().to(self.device) 

    # 			score = self.model(v_p)
    # 			label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)

    # 			if self.binary:
    # 				loss_fct = torch.nn.BCELoss()
    # 				m = torch.nn.Sigmoid()
    # 				n = torch.squeeze(m(score), 1)
    # 				loss = loss_fct(n, label)
    # 			else:
    # 				loss_fct = torch.nn.MSELoss()
    # 				n = torch.squeeze(score, 1)
    # 				loss = loss_fct(n, label)
    # 			loss_history.append(loss.item())

    # 			opt.zero_grad()
    # 			loss.backward()
    # 			opt.step()

    # 			if verbose:
    # 				if (i % 100 == 0):
    # 					t_now = time()
    # 					if verbose:
    # 						print("Training at Epoch " + str(epo + 1) + " iteration " + str(i) + \
    # 						" with loss " + str(loss.cpu().detach().numpy())[:7] +\
    # 						". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
    # 					### record total run time

    # 		##### validate, select the best model up to now 
    # 		with torch.set_grad_enabled(False):
    # 			if self.binary:  
    # 				## binary: ROC-AUC, PR-AUC, F1  
    # 				auc, auprc, f1, logits = self.test_(validation_generator, self.model)
    # 				lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
    # 				valid_metric_record.append(lst)
    # 				if auc > max_auc:
    # 					model_max = copy.deepcopy(self.model)
    # 					max_auc = auc
    # 				if verbose:
    # 					print("Validation at Epoch "+ str(epo + 1) + " , AUROC: " + str(auc)[:7] + \
    # 					  " , AUPRC: " + str(auprc)[:7] + " , F1: "+str(f1)[:7])
    # 			else:  
    # 				### regression: MSE, Pearson Correlation, with p-value, Concordance Index  
    # 				mse, r2, p_val, CI, logits = self.test_(validation_generator, self.model)
    # 				lst = ["epoch " + str(epo)] + list(map(float2str,[mse, r2, p_val, CI]))
    # 				valid_metric_record.append(lst)
    # 				if mse < max_MSE:
    # 					model_max = copy.deepcopy(self.model)
    # 					max_MSE = mse
    # 				if verbose:
    # 					print("Validation at Epoch "+ str(epo + 1) + " , MSE: " + str(mse)[:7] + " , Pearson Correlation: "\
    # 					 + str(r2)[:7] + " with p-value: " + str(f"{p_val:.2E}") +" , Concordance Index: "+str(CI)[:7])
    # 		table.add_row(lst)


    # 	#### after training 
    # 	prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
    # 	with open(prettytable_file, "w") as fp:
    # 		fp.write(table.get_string())

    # 	# load early stopped model
    # 	self.model = model_max

    # 	if test is not None:
    # 		if verbose:
    # 			print("--- Go for Testing ---")
    # 		if self.binary:
    # 			auc, auprc, f1, logits = self.test_(testing_generator, model_max, test = True, verbose = verbose)
    # 			test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
    # 			test_table.add_row(list(map(float2str, [auc, auprc, f1])))
    # 			if verbose:
    # 				print("Testing AUROC: " + str(auc) + " , AUPRC: " + str(auprc) + " , F1: "+str(f1))				
    # 		else:
    # 			mse, r2, p_val, CI, logits = self.test_(testing_generator, model_max, test = True, verbose = verbose)
    # 			test_table = PrettyTable(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
    # 			test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
    # 			if verbose:
    # 				print("Testing MSE: " + str(mse) + " , Pearson Correlation: " + str(r2) 
    # 				  + " with p-value: " + str(f"{p_val:.2E}") +" , Concordance Index: "+str(CI))
    # 		np.save(os.path.join(self.result_folder, str(self.target_encoding)
    # 			     + "_logits.npy"), np.array(logits))                

    # 		######### learning record ###########

    # 		### 1. test results
    # 		prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
    # 		with open(prettytable_file, "w") as fp:
    # 			fp.write(test_table.get_string())

    # 	if verbose:
    # 	### 2. learning curve 
    # 		fontsize = 16
    # 		iter_num = list(range(1,len(loss_history)+1))
    # 		plt.figure(3)
    # 		plt.plot(iter_num, loss_history, "bo-")
    # 		plt.xlabel("iteration", fontsize = fontsize)
    # 		plt.ylabel("loss value", fontsize = fontsize)
    # 		pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
    # 		with open(pkl_file, "wb") as pck:
    # 			pickle.dump(loss_history, pck)

    # 		fig_file = os.path.join(self.result_folder, "loss_curve.png")
    # 		plt.savefig(fig_file)
    # 	if verbose:
    # 		print("--- Training Finished ---")
          

    # def predict(self, df_data, verbose = True):
    # 	"""
    # 		utils.data_process_repurpose_virtual_screening 
    # 		pd.DataFrame
    # 	"""
    # 	if verbose:
    # 		print("predicting...")
    # 	info = data_process_loader_Protein_Prediction(df_data.index.values, df_data.Label.values, df_data, **self.config)
    # 	self.model.to(device)
    # 	params = {"batch_size": self.config["batch_size"],
    # 			"shuffle": False,
    # 			"num_workers": self.config["num_workers"],
    # 			"drop_last": False,
    # 			"sampler":SequentialSampler(info)}

    # 	generator = data.DataLoader(info, **params)

    # 	score = self.test_(generator, self.model, repurposing_mode = True)
    # 	# set repurposong mode to true, will return only the scores.
    # 	return score

    # def save_model(self, path_dir):
    # 	if not os.path.exists(path_dir):
    # 		os.makedirs(path_dir)
    # 	torch.save(self.model.state_dict(), path_dir + "/model.pt")
    # 	save_dict(path_dir, self.config)

    # def load_pretrained(self, path):
    # 	if not os.path.exists(path):
    # 		os.makedirs(path)

    # 	if self.device == "cuda":
    # 		state_dict = torch.load(path)
    # 	else:
    # 		state_dict = torch.load(path, map_location = torch.device("cpu"))
    # 	# to support training from multi-gpus data-parallel:
        
    # 	if next(iter(state_dict))[:7] == "module.":
    # 		# the pretrained model is from data-parallel module
    # 		from collections import OrderedDict
    # 		new_state_dict = OrderedDict()
    # 		for k, v in state_dict.items():
    # 			name = k[7:] # remove `module.`
    # 			new_state_dict[name] = v
    # 		state_dict = new_state_dict

    # 	self.model.load_state_dict(state_dict)

    # 	self.binary = self.config["binary"]


