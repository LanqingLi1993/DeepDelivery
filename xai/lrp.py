# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:32:09 2022

@author: ut
"""
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

from misc import apply_heatmap, get_example_params


class LRP():
    """
        Layer-wise relevance propagation with gamma+epsilon rule

        This code is largely based on the code shared in: https://git.tu-berlin.de/gmontavon/lrp-tutorial
        Some stuff is removed, some stuff is cleaned, and some stuff is re-organized compared to that repository.
    """
    def __init__(self, model):
        self.model = model

    def LRP_forward(self, layer, input_tensor, gamma=None, epsilon=None):
        # This implementation uses both gamma and epsilon rule for all layers
        # The original paper argues that it might be beneficial to sometimes use
        # or not use gamma/epsilon rule depending on the layer location
        # Have a look a the paper and adjust the code according to your needs

        # LRP-Gamma rule
        if gamma is None:
            # gamma = lambda value: value + 0.05 * copy.deepcopy(value.data.detach()).clamp(min=0)
            # gamma = lambda value: value
            gamma = lambda value: value + 0.25 * copy.deepcopy(value.data.detach()).clamp(min=0) # from "Layer‐wise relevance propagation  of InteractionNet explains  protein–ligand interactions at the atom level"
            # gamma = lambda value: value + 100 * copy.deepcopy(value.data.detach()).clamp(min=0)
        # LRP-Epsilon rule
        if epsilon is None:
            # eps = 0
            # eps = 1e-9
            eps = 0.25 # from "Layer‐wise relevance propagation of InteractionNet explains protein–ligand interactions  at the atom level"
            epsilon = lambda value: value + eps

        # Copy the layer to prevent breaking the graph
        layer = copy.deepcopy(layer)

        # Modify weight and bias with the gamma rule
        try:
            layer.weight = nn.Parameter(gamma(layer.weight))
        except AttributeError:
            pass
            # print('This layer has no weight')
        try:
            layer.bias = nn.Parameter(gamma(layer.bias))
        except AttributeError:
            pass
            # print('This layer has no bias')
        # Forward with gamma + epsilon rule
        return epsilon(layer(input_tensor))

    def LRP_step(self, forward_output, layer, LRP_next_layer, gamma=None, epsilon=None):
        if isinstance(layer, nn.AdaptiveAvgPool1d):
            # For AdaptiveMaxPool1d, redistribute the relevance evenly
            return LRP_next_layer.unsqueeze(-1).expand(-1, forward_output.shape[-1])/forward_output.shape[-1]
        elif isinstance(layer, nn.AdaptiveMaxPool1d):
            # Get output size from layer
            out_size = layer.output_size
            
            # Compute max indices using PyTorch's functional API
            _, indices = F.adaptive_max_pool1d(forward_output, 1, return_indices=True)
            
            # Initialize zeros tensor for relevance redistribution
            LRP_this_layer = torch.zeros_like(forward_output)
            
            # Scatter next layer's relevance to max positions

            # LRP_next_layer = LRP_next_layer.double()
            # print(LRP_this_layer.dtype, forward_output.dtype, LRP_next_layer.dtype)
            # print(LRP_this_layer.shape, LRP_next_layer.shape, indices.shape)
            src = LRP_next_layer.unsqueeze(0).unsqueeze(-1).double().expand(-1, -1, LRP_this_layer.shape[-1])
            # print('src', src.shape, src)  
            LRP_this_layer.scatter_(dim=-1, index=indices, src=src)
            # print('LRP_this_layer', LRP_this_layer, LRP_this_layer.nonzero())
            return LRP_this_layer
        # elif isinstance(layer, nn.ReLU):
        #     # # return LRP_next_layer

        #     #######################################################
        #     # Redistribute relevance to positive positions 
        #     #######################################################      
        #     # mask = (forward_output > 0).float()
        #     # return LRP_next_layer * mask

        #     #######################################################
        #     # Bypass ReLU layer 
        #     #######################################################

        #     # return LRP_next_layer

        #     #######################################################
        #     # Redistribute negative relevance to positive positions
        #     #######################################################
        #     epsilon = 1e-9
        #     # Create a mask for positive activations
        #     mask = (forward_output > 0).float()
            
        #     # Compute total relevance assigned to negative positions
        #     negative_relevance = LRP_next_layer * (1 - mask)

        #     total_negative = negative_relevance.sum()
            
        #     # Compute weighted distribution factor for positive positions
        #     # Add epsilon to prevent divide-by-zero
        #     positive_weights = (forward_output * mask) + epsilon
            
        #     # Redistribute negative relevance to positive positions
        #     redistribution = total_negative * (positive_weights / positive_weights.sum())
            
        #     # Combine original positive relevance and redistributed relevance
        #     LRP_this_layer = (LRP_next_layer * mask) + redistribution
        #     # print('relu', LRP_this_layer, len(LRP_this_layer.nonzero()))
            
        #     return LRP_this_layer
        # elif isinstance(layer, nn.Linear):
        #     epsilon = lambda value: value + 1e-3        
        # elif isinstance(layer, nn.Conv1d):
        #     gamma = lambda value: value + 100 * copy.deepcopy(value.data.detach()).clamp(min=0)
        #     epsilon = lambda value: value + 0.25


        # Enable the gradient flow
        forward_output = forward_output.requires_grad_(True)
        # Get LRP forward out based on the LRP rules
        lrp_rule_forward_out = self.LRP_forward(layer, forward_output, gamma=gamma, epsilon=epsilon)
        # Perform element-wise division
        ele_div = (LRP_next_layer / lrp_rule_forward_out).data
        # Propagate
        (lrp_rule_forward_out * ele_div).sum().backward()
        # Get the visualization
        # print('grad', forward_output.grad, len(forward_output.grad.nonzero()))
        LRP_this_layer = (forward_output * forward_output.grad).data
        # if isinstance(layer, nn.AdaptiveAvgPool1d):
        #     print('lrp1', LRP_next_layer.unsqueeze(-1).expand(-1, forward_output.shape[-1])/forward_output.shape[-1])
        #     print('lrp2', LRP_this_layer)
        return LRP_this_layer
        
    def generate(self, input, target_class, target_encoding, num_class, softmax=1):
        assert len(self.model.backbone_kws) == 1, "Only one backbone is currently supported for this LRP implementation"
        forward_output = [input]
        layers_in_model = []

        # for CNN model
        if target_encoding == 'CNN':

            backbone =  getattr(self.model, self.model.backbone_kws[0]) # CNN
            for conv_layer in backbone.conv:
                # forward_output.append(conv_layer(forward_output[-1]).detach())
                # forward_output.append(F.relu(forward_output[-1]).detach())
                forward_output.append(F.relu(conv_layer(forward_output[-1]).detach()))
                # layers_in_model.append(conv_layer)
                # layers_in_model.append(nn.ReLU())
                layers_in_model.append(nn.Sequential(conv_layer, nn.ReLU()))
            forward_output.append(F.adaptive_max_pool1d(forward_output[-1], 1).view(-1).detach()) # for deeppurpose encoders
            # forward_output.append(F.adaptive_avg_pool1d(forward_output[-1], 1).view(-1).detach())

            forward_output.append(backbone.fc1(forward_output[-1].float()).detach())

            layers_in_model.append(nn.AdaptiveMaxPool1d(1))
            # layers_in_model.append(nn.AdaptiveAvgPool1d(1))

            layers_in_model.append(backbone.fc1)
            double_2_float_layer_n = len(layers_in_model) - 1
 
        
            classifier = self.model.head.predictor
            for i, layer in enumerate(classifier):

                if i != len(classifier)-1:
                    forward_output.append(F.relu(layer(forward_output[-1]).detach()))
                    # layers_in_model.append(nn.ReLU())
                    layers_in_model.append(nn.Sequential(layer, nn.ReLU()))
                else:
                    forward_output.append(layer(forward_output[-1]).detach())
                    layers_in_model.append(layer)

            # adding softmax layer
            if softmax:
                forward_output.append(F.softmax(forward_output[-1]).detach())
                layers_in_model.append(nn.Softmax())

        elif target_encoding == 'ESM_embed':
            # input dim: 1 * 1024 * 1280

            forward_output.append(F.adaptive_avg_pool1d(forward_output[-1], 1).view(-1).detach()) 
            layers_in_model.append(nn.AdaptiveAvgPool1d(1))


            backbone =  getattr(self.model, self.model.backbone_kws[0]) # MLP
            for i, layer in enumerate(backbone.predictor):
                forward_output.append(layer(forward_output[-1]).detach())
                layers_in_model.append(layer)
                if i != len(backbone.predictor) -1:
                    forward_output.append(F.relu(forward_output[-1]).detach())
                    layers_in_model.append(nn.ReLU())

            
            classifier = self.model.head.predictor
            for i, layer in enumerate(classifier):
                forward_output.append(layer(forward_output[-1]).detach())
                layers_in_model.append(layer)
                if i != len(classifier)-1:
                    forward_output.append(F.relu(forward_output[-1]).detach())
                    layers_in_model.append(nn.ReLU())

            # adding softmax layer
            if softmax:
                forward_output.append(F.softmax(forward_output[-1]).detach())
                layers_in_model.append(nn.Softmax())

        elif target_encoding == 'Transformer':
            backbone = getattr(self.model, self.model.backbone_kws[0]) # Transformer
            # embed = backbone(forward_output[-1], pooling=0) 
            # print(embed[0][1].shape, embed[1][0].shape)

            forward_output.append(torch.tensor(backbone(forward_output[-1], pooling=0)[0]).detach()) # n_batch * hidden_dim * seq_len
            layers_in_model.append(backbone)
            forward_output.append(F.adaptive_avg_pool1d(forward_output[-1], 1).view(1, -1).detach())
            layers_in_model.append(nn.AdaptiveAvgPool1d(1))

            classifier = self.model.head.predictor
            for i, layer in enumerate(classifier):
                # print(forward_output[-1].shape)
                forward_output.append(layer(forward_output[-1]).detach())
                layers_in_model.append(layer)
                if i != len(classifier)-1:
                    forward_output.append(F.relu(forward_output[-1]).detach())
                    layers_in_model.append(nn.ReLU())

            # adding softmax layer
            if softmax:
                forward_output.append(F.softmax(forward_output[-1]).detach())
                layers_in_model.append(nn.Softmax())
        else:
            NotImplementedError
        number_of_layers = len(layers_in_model)        
        # Target for backprop
        target_class_one_hot = torch.FloatTensor(1, num_class).zero_()
        target_class_one_hot[0][target_class] = 1

        # This is where we accumulate the LRP results
        LRP_per_layer = [None] * number_of_layers + [(forward_output[-1] * target_class_one_hot).data]

        for layer_index in range(number_of_layers)[::-1]:

            # if isinstance(layers_in_model[layer_index], (nn.Linear, nn.Conv1d, nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d)): # bypass non-linear layers such as ReLU, Softmax, as we can interpret them as a normalization layer
            if isinstance(layers_in_model[layer_index], (nn.Linear, nn.Conv1d, nn.Sequential, nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d, nn.Softmax)):
                # In the paper implementation, they replace maxpool with avgpool because of certain properties
                # I didn't want to modify the model like the original implementation but
                # feel free to modify this part according to your need(s)
                if target_encoding == 'CNN':
                    # print('layer', type(layers_in_model[layer_index]))
                    if layer_index == double_2_float_layer_n:

                        lrp_this_layer = self.LRP_step(forward_output[layer_index].float(), layers_in_model[layer_index], LRP_per_layer[layer_index+1])
                    else:
                        # print(double_2_float_layer_n, layer_index, forward_output[layer_index], layers_in_model[layer_index])
                        # print(forward_output[layer_index].shape, LRP_per_layer[layer_index+1].shape, layers_in_model[layer_index])

                        lrp_this_layer = self.LRP_step(forward_output[layer_index], layers_in_model[layer_index], LRP_per_layer[layer_index+1])
                    # print('lrp_this_layer', lrp_this_layer.shape, lrp_this_layer, len(lrp_this_layer.nonzero()))
                else:
                    lrp_this_layer = self.LRP_step(forward_output[layer_index], layers_in_model[layer_index], LRP_per_layer[layer_index+1])

                LRP_per_layer[layer_index] = lrp_this_layer
            else:
                LRP_per_layer[layer_index] = LRP_per_layer[layer_index+1]
        return LRP_per_layer

    def generate_old(self, input_image, target_class):
        layers_in_model = list(self.model._modules['features']) + list(self.model._modules['classifier'])
        number_of_layers = len(layers_in_model)
        # Needed to know where flattening happens
        features_to_classifier_loc = len(self.model._modules['features'])

        # Forward outputs start with the input image
        forward_output = [input_image]
        # Then we do forward pass with each layer
        for conv_layer in list(self.model._modules['features']):
            forward_output.append(conv_layer.forward(forward_output[-1].detach()))

        # To know the change in the dimensions between features and classifier
        feature_to_class_shape = forward_output[-1].shape
        # Flatten so we can continue doing forward passes at classifier layers
        forward_output[-1] = torch.flatten(forward_output[-1], 1)
        for index, classifier_layer in enumerate(list(self.model._modules['classifier'])):
            forward_output.append(classifier_layer.forward(forward_output[-1].detach()))

        # Target for backprop
        target_class_one_hot = torch.FloatTensor(1, 1000).zero_()
        target_class_one_hot[0][target_class] = 1

        # This is where we accumulate the LRP results
        LRP_per_layer = [None] * number_of_layers + [(forward_output[-1] * target_class_one_hot).data]

        for layer_index in range(1, number_of_layers)[::-1]:
            # This is where features to classifier change happens
            # Have to flatten the lrp of the next layer to match the dimensions
            if layer_index == features_to_classifier_loc-1:
                LRP_per_layer[layer_index+1] = LRP_per_layer[layer_index+1].reshape(feature_to_class_shape)

            if isinstance(layers_in_model[layer_index], (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MaxPool2d)):
                # In the paper implementation, they replace maxpool with avgpool because of certain properties
                # I didn't want to modify the model like the original implementation but
                # feel free to modify this part according to your need(s)
                lrp_this_layer = self.LRP_step(forward_output[layer_index], layers_in_model[layer_index], LRP_per_layer[layer_index+1])
                LRP_per_layer[layer_index] = lrp_this_layer
            else:
                LRP_per_layer[layer_index] = LRP_per_layer[layer_index+1]
        return LRP_per_layer

def lrp_score(model, i_layer, target_encoding, num_class, target_class, df, output_dir, key_lst, origin_key_lst, normalize=1, max_seq_len=1024, bpe_encoding=0, offset_lst=[0, 3, 3, 10, 10, 21, 21], 
              softmax=1,
              targetclassonly=0):
        '''
        Calculate the LRP score for each domain/subclass and store them in a table

        Arguments:
            targetclassonly: only compute samples belonging to the target class for total LRP score
        '''
        import _init_paths
        from DeepPurpose.utils import trans_protein, protein_2_embed
        from utils.featurizer import protein2emb_encoder, protein2emb_encoder_custom
        from misc import get_segment, get_seq

        new_df = pd.DataFrame(columns=df.columns.tolist() + ['rest'])
        layerwise_relevance = LRP(model)
        total_lrp_score = {}
        total_lrp_count = {}
        total_partition_score = 0
        for key in new_df.columns.tolist():
            if key not in key_lst: 
                total_lrp_score[key] = []
                total_lrp_count[key] = []

        for idx, row in df.iterrows():
            new_row = {}
            partition_row = {}
            for key in origin_key_lst:
                new_row[key] = [row[key]]

            # new_row = {'Entry': [row['Entry']], 
            #     'Entry Name': [row['Entry Name']],
            #     'X': [row['X']], 'Subclass': [row['Subclass']]}
            # partition_row = {'Entry': [row['Entry']], 
            #     'Entry Name': [row['Entry Name']],
            #     'X': [row['X']], 'Subclass': [row['Subclass']]}
            input = row['X']
            # input = 'MDGSGPFSCPICLEPLREPVTLPCGHNFCLACLGALWPHRSAGGTGGSGGPARCPLCQEPFPDGLQLRKNHTLSELLQLRQGSVPGPMSAPASGSTRGATPEPSAPSAPPPAPEPSAPCAPEQWPAGEEPVRCDACPEGAALPAALSCLSCLASFCSAHLAPHERSPALRGHRLVPPLRRLEESLCPRHLRPLERYCRVERVCLCEACATQDHRGHELVPLEQERALQEVEQSKVLSAAEDRMDELGAGIAQSRRTVALIKSAAVAERERVSQMFAEATATLQSFQNEVMGFIEEGEATMLGRSQGDLRRQEEQRSRLSKARHNLGQVPEADSVSFLQELLALRLALEEGCGPGPGPPRELSFTKSSQVVKAVRDTLISACASQWEQLRGLGSNEDGLQKLGSEDVESQDPDSTSLLESEAPRDYFLKFAYIVDLDSDTADKFLQLFGTKGVKRVLCPINYPESPTRFTHCEQVLGEGALDRGTYYWEVEIIEGWVSVGVMAEGFSPQEPYDRGRLGRNAHSCCLQWNGRGFSVWFCGLEAPLPHAFSPTVGVCLEYADHALAFYAVRDGKLSLLRRLKASRPRRSGALASPTDPFQSRLDSHFSGLFNHRLKPAFFLESVDAHLQIGPLKKSCITVLKRR'
            input_len = len(input)
            # print(protein_2_embed(trans_protein(input))[0, :])

            if target_encoding == "CNN":
                prep_input = torch.tensor(protein_2_embed(trans_protein(input))).double()
                prep_input = prep_input.unsqueeze(0)
                new_row['Predicted Score'] =  [F.softmax(model([prep_input])[0][0]).detach().numpy()]
            elif target_encoding == "ESM_embed":
                prep_input = torch.tensor(input) # 1 * 1024 * 1280
                prep_input = prep_input.unsqueeze(0).transpose(1, 2) # 1 * 1280 * 1024
                new_row['Predicted Score'] =  [F.softmax(model([prep_input[:, :, -1]])[0][0]).detach().numpy()]
            elif target_encoding == "Transformer":
                if bpe_encoding:
                    prep_input = protein2emb_encoder(input, max_seq_len)
                else:                    
                    prep_input = protein2emb_encoder_custom(input, max_seq_len)
                prep_input = torch.stack([torch.tensor(i).unsqueeze(0) for i in prep_input], axis=1) # bs * 2 * seq_len
                output = model([prep_input])
                new_row['Predicted Score'] = [np.array(F.softmax(output[0][0]).detach().numpy())]
                
            else:
                NotImplementedError


            # Generate visualization(s)
            LRP_per_layer = layerwise_relevance.generate(prep_input, target_class=target_class, target_encoding=target_encoding, num_class=num_class, softmax=softmax)


            if target_encoding == "CNN":
                # seq_len = 1000 # max length of the sequence for plotting
                if input_len > 979:
                    pad_lrp_layer = np.zeros(max_seq_len)
                    LRP_layer_len = LRP_per_layer[i_layer][0].shape[-1]
                    pad_lrp_layer[:LRP_layer_len] = LRP_per_layer[i_layer][0].sum(axis=0)[:LRP_layer_len]


                else:
                    pad_lrp_layer = np.array(LRP_per_layer[i_layer][0].sum(axis=0))
            elif target_encoding == "ESM_embed":
                # seq_len = 1024
                if input_len > max_seq_len:
                    pad_lrp_layer = np.zeros(max_seq_len)
                    LRP_layer_len = LRP_per_layer[i_layer].shape[-1]
                    pad_lrp_layer[:LRP_layer_len] = LRP_per_layer[i_layer][0].sum(axis=0)[:LRP_layer_len]

                else:
                    pad_lrp_layer = np.array(LRP_per_layer[i_layer][0].sum(axis=0))
            elif target_encoding == "Transformer":
                if input_len > max_seq_len:
                    pad_lrp_layer = np.zeros(max_seq_len)
                    LRP_layer_len = LRP_per_layer[i_layer].shape[-1]
                    pad_lrp_layer[:LRP_layer_len] = LRP_per_layer[i_layer][0].sum(axis=0)[:LRP_layer_len]

                else:
                    pad_lrp_layer = np.array(LRP_per_layer[i_layer][0].sum(axis=0))

            else:
                NotImplementedError

            new_row['lrp_score'] = [pad_lrp_layer]
            # seg_threshold = np.percentile(np.abs(pad_lrp_layer), 95)/2
            seg_threshold = .5/input_len
            seg_mask = (pad_lrp_layer > seg_threshold)
            lrp_segments = get_segment(seg_mask, offset=offset_lst[i_layer], max_idx=pad_lrp_layer.size)
            new_row['lrp_segments'] = [lrp_segments]

            if target_encoding == "ESM_embed":
                new_row['lrp_aa_segments'] = [get_seq(row['X_seq'], lrp_segments)]
            elif target_encoding == "CNN":
                new_row['lrp_aa_segments'] = [get_seq(row['X'], lrp_segments)]
            elif target_encoding == "Transformer":
                new_row['lrp_aa_segments'] = [get_seq(row['X'], lrp_segments)]
            # print('total score:', pad_lrp_layer.sum())
            # print(np.histogram(pad_lrp_layer))
            # print(np.sum(np.ones_like(pad_lrp_layer)))
            # threshold = np.percentile(np.abs(pad_lrp_layer), 97)/5
            # print(np.sum(np.abs(pad_lrp_layer) > threshold))
            # print(np.sum(pad_lrp_layer != 0))
            # threshold: average of top 3% divided by 10
            lrp_mask = np.ones_like(pad_lrp_layer)

            rowise_partition_score = float(pad_lrp_layer.sum())

            if normalize:
                norm_factor = np.abs(float(pad_lrp_layer.sum()))/lrp_mask.sum()
                for key in df.columns.tolist():
                    if key not in key_lst:
                        lrp_key_mask = np.zeros_like(lrp_mask) # lrp mask for each domain
                        lrp_score = 0
                        lrp_count = 0
                        lrp_segments = row[key]
                        for segment in lrp_segments:
                            lrp_mask[segment[0]-1:segment[1]] = 0
                            lrp_key_mask[segment[0]-1:segment[1]] = 1
                        lrp_score += (pad_lrp_layer * lrp_key_mask).sum()
                        lrp_count += lrp_key_mask.sum()

                        # total_lrp_score[key].append(np.abs(float(lrp_score))/norm_factor)
                        # total_lrp_count[key].append(int(lrp_count))
                        if targetclassonly:
                            # print('Predicted Score:', new_row['Predicted Score'][0])
                            if np.argmax(np.array(new_row['Predicted Score'][0])) == target_class: # Predicted Score: [array([0.9986204 , 0.00137962], dtype=float32)]
                                total_lrp_score[key].append(float(lrp_score)/norm_factor)
                                total_lrp_count[key].append(int(lrp_count))
                            else:
                                total_lrp_score[key].append(0)
                                total_lrp_count[key].append(0)
                        else:
                            total_lrp_score[key].append(float(lrp_score)/norm_factor)
                            total_lrp_count[key].append(int(lrp_count))
                        # new_row[key] = np.abs(float(lrp_score))/(lrp_count+1e-8)/norm_factor
                        new_row[key] = float(lrp_score)/(lrp_count+1e-8)/norm_factor

                        partition_row[key] = float(lrp_score)/(rowise_partition_score+1e-8)
                total_partition_score += rowise_partition_score
                rest_score = (pad_lrp_layer * lrp_mask).sum()
                rest_count = lrp_mask.sum()
                # new_row['rest'] = np.abs(float(rest_score)) / (rest_count + 1e-8) / norm_factor # for the rest of the sequence
                new_row['rest'] = float(rest_score) / (rest_count + 1e-8) / norm_factor # for the rest of the sequence

                partition_row['rest'] = float(rest_score) / (rowise_partition_score + 1e-8)
                # total_lrp_score['rest'].append(np.abs(float(rest_score))/norm_factor)

                if targetclassonly:
                    if np.argmax(np.array(new_row['Predicted Score'][0])) == target_class:
                        total_lrp_score['rest'].append(float(rest_score)/norm_factor)
                        total_lrp_count['rest'].append(int(rest_count))
                    else:
                        total_lrp_score['rest'].append(0)
                        total_lrp_count['rest'].append(0)
                else:
                    total_lrp_score['rest'].append(float(rest_score)/norm_factor)
                    total_lrp_count['rest'].append(int(rest_count))

                new_df = pd.concat([new_df, pd.DataFrame(new_row)], ignore_index=True)
            else:
                for key in df.columns.tolist():
                    if key not in key_lst:
                        key_mask = np.zeros_like(lrp_mask)
                        lrp_score = 0
                        lrp_count = 0
                        segments = row[key]
                        for segment in segments:
                            lrp_mask[segment[0]-1:segment[1]] = 0
                            key_mask[segment[0]-1:segment[1]] = 1
                        lrp_score += (pad_lrp_layer * key_mask).sum()
                        lrp_count += key_mask.sum()
                        # total_lrp_score[key].append(np.abs(float(lrp_score)))
                        if targetclassonly:
                            # print('Predicted Score:', new_row['Predicted Score'][0])
                            if np.argmax(np.array(new_row['Predicted Score'][0])) == target_class:
                                total_lrp_score[key].append(float(lrp_score))
                                total_lrp_count[key].append(int(lrp_count))
                            else:
                                total_lrp_score[key].append(0)
                                total_lrp_count[key].append(0)
                        else:
                            total_lrp_score[key].append(float(lrp_score))
                            total_lrp_count[key].append(int(lrp_count))
                        # new_row[key] = np.abs(float(lrp_score))/(lrp_count+1e-8)
                        new_row[key] = float(lrp_score)/(lrp_count+1e-8)
                        partition_row[key] = float(lrp_score)/(rowise_partition_score+1e-8)
                total_partition_score += rowise_partition_score
                rest_score = (pad_lrp_layer * lrp_mask).sum()
                rest_count = lrp_mask.sum()
                # new_row['rest'] = np.abs(float(rest_score)) / (rest_count + 1e-8) # for the rest of the sequence
                new_row['rest'] = float(rest_score) / (rest_count + 1e-8) # for the rest of the sequence

                partition_row['rest'] = float(rest_score) / (rowise_partition_score + 1e-8)
                # total_lrp_score['rest'].append(np.abs(float(rest_score)))

                if targetclassonly:
                    if np.argmax(np.array(new_row['Predicted Score'][0])) == target_class:
                        total_lrp_score['rest'].append(float(rest_score))
                        total_lrp_count['rest'].append(int(rest_count))
                    else:
                        total_lrp_score['rest'].append(0)
                        total_lrp_count['rest'].append(0)
                else:
                    total_lrp_score['rest'].append(float(rest_score))
                    total_lrp_count['rest'].append(int(rest_count))
                new_df = pd.concat([new_df, pd.DataFrame(new_row)], ignore_index=True)
                # new_df = pd.concat([new_df, pd.DataFrame(partition_row)], ignore_index=True)

        for k, _ in total_lrp_score.items():
            total_lrp_score[k] = np.array(total_lrp_score[k])
        for k, _ in total_lrp_count.items():
            total_lrp_count[k] = np.array(total_lrp_count[k])
        total_lrp = {}
        total_partition = {}

        for key in new_df.columns.tolist():
            if key not in key_lst:
                total_lrp[key] = total_lrp_score[key].sum()/(total_lrp_count[key].sum()+1e-8)
                # total_partition[key] = float(total_lrp_score[key])/(total_partition_score+1e-8)

        total_lrp_lst = [(k, v) for k, v in total_lrp.items()]
        total_lrp_lst = sorted(total_lrp_lst, key=lambda x: x[target_class], reverse=True)
        for total_lrp_item in total_lrp_lst:
            print(total_lrp_item)

    
        # calculate the LRP score for each subclass
        subclass_lst = new_df['Subclass'].unique()
        for subclass in subclass_lst:
            print(subclass)
            if targetclassonly:
                mask = (new_df['Subclass'] == subclass) & (new_df['Predicted Score'].apply(lambda x: np.argmax(np.array(x)) == target_class)).values
            else:
                mask = (new_df['Subclass'] == subclass).values
            subclass_lrp = {}
            
            for key in new_df.columns.tolist():
                if key not in key_lst:
                    subclass_lrp[key] = (total_lrp_score[key] * mask).sum()/((total_lrp_count[key] * mask).sum()+1e-8)
                    # total_partition[key] = float(total_lrp_score[key])/(total_partition_score+1e-8)

            subclass_lrp_lst = [(k, v) for k, v in subclass_lrp.items()]
            subclass_lrp_lst = sorted(subclass_lrp_lst, key=lambda x: x[target_class], reverse=True)
            for subclass_lrp_item in subclass_lrp_lst:
                print(subclass_lrp_item)
        
        new_df = pd.concat([new_df, pd.DataFrame(total_lrp, index=[-1])], ignore_index=True)
        # new_df = pd.concat([new_df, pd.DataFrame(total_partition, index=[-1])], ignore_index=True)


        output_path = os.path.join(output_dir, 'lrp_scores-layer{}-targetclass{}-normalize{}-softmax{}-targetclassonly{}-v6.pkl'.format(i_layer, target_class, normalize, softmax, targetclassonly))
        new_df.to_pickle(output_path)

def att_score(model, i_layer, target_encoding, target_class, df, output_dir, key_lst, origin_key_lst, normalize=1, max_seq_len=1024, 
              bpe_encoding=1,
              targetclassonly=0): # for transformer-based model only 
        '''
        Calculate the attention score for each domain/subclass and store them in a table
        '''

        import _init_paths
        from utils.featurizer import protein2emb_encoder, protein2emb_encoder_custom
        from misc import get_segment, get_seq

        new_df = pd.DataFrame(columns=df.columns.tolist() + ['rest'])
        total_att_score = {}
        total_att_count = {}
        total_partition_score = 0

        for key in new_df.columns.tolist():
            if key not in key_lst: 
                total_att_score[key] = []
                total_att_count[key] = []

        for idx, row in df.iterrows():
            new_row = {}
            partition_row = {}
            for key in origin_key_lst:
                new_row[key] = [row[key]]

            input = row['X']
            # input = 'MDGSGPFSCPICLEPLREPVTLPCGHNFCLACLGALWPHRSAGGTGGSGGPARCPLCQEPFPDGLQLRKNHTLSELLQLRQGSVPGPMSAPASGSTRGATPEPSAPSAPPPAPEPSAPCAPEQWPAGEEPVRCDACPEGAALPAALSCLSCLASFCSAHLAPHERSPALRGHRLVPPLRRLEESLCPRHLRPLERYCRVERVCLCEACATQDHRGHELVPLEQERALQEVEQSKVLSAAEDRMDELGAGIAQSRRTVALIKSAAVAERERVSQMFAEATATLQSFQNEVMGFIEEGEATMLGRSQGDLRRQEEQRSRLSKARHNLGQVPEADSVSFLQELLALRLALEEGCGPGPGPPRELSFTKSSQVVKAVRDTLISACASQWEQLRGLGSNEDGLQKLGSEDVESQDPDSTSLLESEAPRDYFLKFAYIVDLDSDTADKFLQLFGTKGVKRVLCPINYPESPTRFTHCEQVLGEGALDRGTYYWEVEIIEGWVSVGVMAEGFSPQEPYDRGRLGRNAHSCCLQWNGRGFSVWFCGLEAPLPHAFSPTVGVCLEYADHALAFYAVRDGKLSLLRRLKASRPRRSGALASPTDPFQSRLDSHFSGLFNHRLKPAFFLESVDAHLQIGPLKKSCITVLKRR'
            input_len = len(input)
            # print(protein_2_embed(trans_protein(input))[0, :])

            if target_encoding == "Transformer": # WARNING: be careful with BPE tokenization, which should be mapped back to the original amino acids space
                if bpe_encoding:
                    prep_input = protein2emb_encoder(input, max_seq_len)
                else:                    
                    prep_input = protein2emb_encoder_custom(input, max_seq_len)
                prep_input = torch.stack([torch.tensor(i).unsqueeze(0) for i in prep_input], axis=1)
                output = model([prep_input])
                new_row['Predicted Score'] = [np.array(F.softmax(output[0][0]).detach().numpy())]
                att_per_layer = [i.detach().numpy() for i in output[1][0]] # n_layer * bs * n_head * n_seq * n_seq
            else:
                NotImplementedError

            if input_len > max_seq_len:
                pad_att_layer = np.zeros(max_seq_len)
                att_layer_len = att_per_layer[i_layer].shape[-1]
                pad_att_layer[:att_layer_len] = att_per_layer[i_layer][0].sum(axis=(0, 1))[:att_layer_len] # sum over all heads and tokens

            else:
                pad_att_layer = np.array(att_per_layer[i_layer][0].sum(axis=(0, 1)))


            new_row['att_score'] = [pad_att_layer]
            seg_threshold = np.percentile(np.abs(pad_att_layer), 95)/2
            seg_mask = (np.abs(pad_att_layer) > seg_threshold)
            att_segments = get_segment(seg_mask, max_idx=pad_att_layer.size)
            new_row['att_segments'] = [att_segments]

            if target_encoding == "Transformer":
                new_row['att_aa_segments'] = [get_seq(row['X'], att_segments)]
            else:
                NotImplementedError

            att_mask = np.ones_like(pad_att_layer)

            rowise_partition_score = float(pad_att_layer.sum())

            if normalize:
                norm_factor = np.abs(float(pad_att_layer.sum()))/mask.sum()
                for key in df.columns.tolist():
                    if key not in key_lst:
                        att_key_mask = np.zeros_like(mask) # attention mask for each domain
                        att_score = 0
                        att_count = 0
                        att_segments = row[key]
                        for segment in att_segments:
                            att_mask[segment[0]-1:segment[1]] = 0
                            att_key_mask[segment[0]-1:segment[1]] = 1
                        att_score += (pad_att_layer * att_key_mask).sum()
                        att_count += att_key_mask.sum()

                        # total_lrp_score[key].append(np.abs(float(lrp_score))/norm_factor)
                        # total_lrp_count[key].append(int(lrp_count))
                        if targetclassonly:
                            # print('Predicted Score:', new_row['Predicted Score'][1])
                            if np.argmax(np.array(new_row['Predicted Score'][0])) == target_class: # Predicted Score: [array([0.9986204 , 0.00137962], dtype=float32)]
                                total_att_score[key].append(float(att_score)/norm_factor)
                                total_att_count[key].append(int(att_count))
                            else:
                                total_att_score[key].append(0)
                                total_att_count[key].append(0)
                        else:
                            total_att_score[key].append(float(att_score)/norm_factor)
                            total_att_count[key].append(int(att_count))

                        # new_row[key] = np.abs(float(lrp_score))/(lrp_count+1e-8)/norm_factor
                        new_row[key] = float(att_score)/(att_count+1e-8)/norm_factor

                        partition_row[key] = float(att_score)/(rowise_partition_score+1e-8)
                total_partition_score += rowise_partition_score
                rest_score = (pad_att_layer * att_mask).sum()
                rest_count = att_mask.sum()
                # new_row['rest'] = np.abs(float(rest_score)) / (rest_count + 1e-8) / norm_factor # for the rest of the sequence
                new_row['rest'] = float(rest_score) / (rest_count + 1e-8) / norm_factor # for the rest of the sequence

                partition_row['rest'] = float(rest_score) / (rowise_partition_score + 1e-8)
                # total_lrp_score['rest'].append(np.abs(float(rest_score))/norm_factor)

                
                if targetclassonly:
                    if np.argmax(np.array(new_row['Predicted Score'][0])) == target_class: 
                        total_att_score['rest'].append(float(rest_score)/norm_factor)
                        total_att_count['rest'].append(int(rest_count))
                    else:
                        total_att_score['rest'].append(0)
                        total_att_count['rest'].append(0)
                else:
                    total_att_score['rest'].append(float(rest_score)/norm_factor)
                    total_att_count['rest'].append(int(rest_count))

                new_df = pd.concat([new_df, pd.DataFrame(new_row)], ignore_index=True)
            else:
                for key in df.columns.tolist():
                    if key not in key_lst:
                        att_key_mask = np.zeros_like(att_mask)
                        att_score = 0
                        att_count = 0
                        segments = row[key]
                        for segment in segments:
                            att_mask[segment[0]-1:segment[1]] = 0
                            att_key_mask[segment[0]-1:segment[1]] = 1
                        att_score += (pad_att_layer * att_key_mask).sum()
                        att_count += att_key_mask.sum()
                        # total_lrp_score[key].append(np.abs(float(lrp_score)))
                        if targetclassonly:
                            # print('Predicted Score:', new_row['Predicted Score'][1])
                            if np.argmax(np.array(new_row['Predicted Score'][0])) == target_class:  # Predicted Score: [array([0.9986204 , 0.00137962], dtype=float32)]
                                total_att_score[key].append(float(att_score))
                                total_att_count[key].append(int(att_count))
                            else:
                                total_att_score[key].append(0)
                                total_att_count[key].append(0)
                        else:
                            total_att_score[key].append(float(att_score))
                            total_att_count[key].append(int(att_count))
                        # new_row[key] = np.abs(float(lrp_score))/(lrp_count+1e-8)
                        new_row[key] = float(att_score)/(att_count+1e-8)
                        partition_row[key] = float(att_score)/(rowise_partition_score+1e-8)
                total_partition_score += rowise_partition_score
                rest_score = (pad_att_layer * att_mask).sum()
                rest_count = att_mask.sum()
                # new_row['rest'] = np.abs(float(rest_score)) / (rest_count + 1e-8) # for the rest of the sequence
                new_row['rest'] = float(rest_score) / (rest_count + 1e-8) # for the rest of the sequence

                partition_row['rest'] = float(rest_score) / (rowise_partition_score + 1e-8)
                # total_lrp_score['rest'].append(np.abs(float(rest_score)))

                if targetclassonly:
                    if np.argmax(np.array(new_row['Predicted Score'][0])) == target_class: 
                        total_att_score['rest'].append(float(rest_score))
                        total_att_count['rest'].append(int(rest_count))
                    else:
                        total_att_score['rest'].append(0)
                        total_att_count['rest'].append(0)
                else:
                    total_att_score['rest'].append(float(rest_score))
                    total_att_count['rest'].append(int(rest_count))
                new_df = pd.concat([new_df, pd.DataFrame(new_row)], ignore_index=True)
                # new_df = pd.concat([new_df, pd.DataFrame(partition_row)], ignore_index=True)

        for k, _ in total_att_score.items():
            total_att_score[k] = np.array(total_att_score[k])
        for k, _ in total_att_count.items():
            total_att_count[k] = np.array(total_att_count[k])
        total_att = {}
        total_partition = {}

        for key in new_df.columns.tolist():
            if key not in key_lst:
                total_att[key] = total_att_score[key].sum()/(total_att_count[key].sum()+1e-8)
                # total_partition[key] = float(total_lrp_score[key])/(total_partition_score+1e-8)

        total_att_lst = [(k, v) for k, v in total_att.items()]
        total_att_lst = sorted(total_att_lst, key=lambda x: x[target_class], reverse=True)
        for total_att_item in total_att_lst:
            print(total_att_item)

    
        # calculate the attention score for each subclass
        subclass_lst = new_df['Subclass'].unique()
        for subclass in subclass_lst:
            print(subclass)
            if targetclassonly:
                mask = (new_df['Subclass'] == subclass) & (new_df['Predicted Score'].apply(lambda x: np.argmax(np.array(x)) == target_class)).values
            else:
                mask = (new_df['Subclass'] == subclass).values
            subclass_att = {}
            
            for key in new_df.columns.tolist():
                if key not in key_lst:
                    subclass_att[key] = (total_att_score[key] * mask).sum()/((total_att_count[key] * mask).sum()+1e-8)
                    # total_partition[key] = float(total_lrp_score[key])/(total_partition_score+1e-8)

            subclass_att_lst = [(k, v) for k, v in subclass_att.items()]
            subclass_att_lst = sorted(subclass_att_lst, key=lambda x: x[target_class], reverse=True)
            for subclass_att_item in subclass_att_lst:
                print(subclass_att_item)
        
        new_df = pd.concat([new_df, pd.DataFrame(total_att, index=[-1])], ignore_index=True)


        output_path = os.path.join(output_dir, 'att_scores-layer{}-targetclass{}-normalize{}-targetclassonly{}-v6.pkl'.format(i_layer, target_class, normalize, targetclassonly))
        new_df.to_pickle(output_path)


def visualize_lrp(model, df, target_encoding, inputs, max_seq_len=1024, bpe_encoding=1):
    IDs = df['ID']
    seqs = df['X']
    ccs = df['cc'] # coiled coil
    dms = df['dm'] # domain
    zfs = df['zf'] # zinc finger
    rgs = df['rg'] # region

    for id, input, cc, dm, zf, rg, seq in zip(IDs, inputs, ccs, dms, zfs, rgs, seqs):
        # input = 'MDGSGPFSCPICLEPLREPVTLPCGHNFCLACLGALWPHRSAGGTGGSGGPARCPLCQEPFPDGLQLRKNHTLSELLQLRQGSVPGPMSAPASGSTRGATPEPSAPSAPPPAPEPSAPCAPEQWPAGEEPVRCDACPEGAALPAALSCLSCLASFCSAHLAPHERSPALRGHRLVPPLRRLEESLCPRHLRPLERYCRVERVCLCEACATQDHRGHELVPLEQERALQEVEQSKVLSAAEDRMDELGAGIAQSRRTVALIKSAAVAERERVSQMFAEATATLQSFQNEVMGFIEEGEATMLGRSQGDLRRQEEQRSRLSKARHNLGQVPEADSVSFLQELLALRLALEEGCGPGPGPPRELSFTKSSQVVKAVRDTLISACASQWEQLRGLGSNEDGLQKLGSEDVESQDPDSTSLLESEAPRDYFLKFAYIVDLDSDTADKFLQLFGTKGVKRVLCPINYPESPTRFTHCEQVLGEGALDRGTYYWEVEIIEGWVSVGVMAEGFSPQEPYDRGRLGRNAHSCCLQWNGRGFSVWFCGLEAPLPHAFSPTVGVCLEYADHALAFYAVRDGKLSLLRRLKASRPRRSGALASPTDPFQSRLDSHFSGLFNHRLKPAFFLESVDAHLQIGPLKKSCITVLKRR'
        input_len = len(seq)
        # print(protein_2_embed(trans_protein(input))[0, :])
        # prep_input = torch.tensor(protein_2_embed(trans_protein(input))).double()


        if target_encoding == "CNN":
            prep_input = torch.tensor(protein_2_embed(trans_protein(input))).double()

        elif target_encoding == "ESM_embed":
            prep_input = torch.tensor(input) # 1 * 1024 * 1280
            prep_input = prep_input.unsqueeze(0).transpose(1, 2) # 1 * 1280 * 1024

        elif target_encoding == "Transformer":
            if bpe_encoding:
                    prep_input = protein2emb_encoder(input, max_seq_len)
            else:                    
                    prep_input = protein2emb_encoder_custom(input, max_seq_len)
            # prep_input = (torch.tensor(prep_input[0].reshape(1, -1)), torch.tensor(prep_input[1].reshape(1, -1))) # seq, mask
            prep_input = torch.stack([torch.tensor(i).unsqueeze(0) for i in prep_input], axis=1) # bs * 2 * seq_len
        else:
            NotImplementedError


        layerwise_relevance = LRP(model)

        # Generate visualization(s)
        num_class = 2
        target_class = 1
        softmax = 1
        LRP_per_layer = layerwise_relevance.generate(prep_input, target_class=target_class, target_encoding=target_encoding, num_class=num_class, softmax=softmax)
        for i, lrp in enumerate(LRP_per_layer):
            print('layer{}'.format(i), lrp.shape)

        lrp_to_vis = []


        if target_encoding == "CNN":
            if input_len > 979:
                # #####
                # # for visualizing 1D LRP score
                # #####
                for i_layer in range(7):
                    pad_lrp_layer = np.zeros(max_seq_len)
                    LRP_layer_len = LRP_per_layer[i_layer].shape[-1]
                    pad_lrp_layer[:LRP_layer_len] = LRP_per_layer[i_layer].sum(axis=0)[:LRP_layer_len]
                    lrp_to_vis.append(pad_lrp_layer)
                base_seq = np.zeros((4, max_seq_len)) # for visualizing protein functional domains

                #####
                # for visualizing 2D LRP score
                #####
                # pad_lrp_layer = np.zeros((LRP_per_layer[i_layer][0].shape[0], max_seq_len))
                # LRP_layer_len = LRP_per_layer[i_layer].shape[-1]
                # pad_lrp_layer[:, :LRP_layer_len] = LRP_per_layer[i_layer][:, :LRP_layer_len]
                # lrp_to_vis += list(pad_lrp_layer)
                # base_seq = np.zeros((4, seq_len)) # for visualizing protein functional domains



            else:
                # #####
                # # for visualizing 1D LRP score
                # #####
                for i_layer in range(7):
                    lrp_to_vis.append(LRP_per_layer[i_layer].sum(axis=0)[:input_len].numpy())
                base_seq = np.zeros((4, input_len))



                #####
                # for visualizing 2D LRP score
                #####

                # lrp_to_vis += list(LRP_per_layer[i_layer][:, :input_len].numpy())
                # base_seq = np.zeros((4, input_len))
                # seq_len = input_len  

        elif target_encoding == "ESM_embed":
            i_layer = 0
            if input_len > max_seq_len:
                pad_lrp_layer = np.zeros(max_seq_len)
                LRP_layer_len = LRP_per_layer[i_layer].shape[-1]
                pad_lrp_layer[:LRP_layer_len] = LRP_per_layer[i_layer][0].sum(axis=0)[:LRP_layer_len]
                lrp_to_vis.append(pad_lrp_layer)
                base_seq = np.zeros((4, max_seq_len)) # for visualizing protein functional domains
            else:
                lrp_to_vis.append(LRP_per_layer[i_layer][0].sum(axis=0)[:input_len].numpy())
                base_seq = np.zeros((4, input_len))


        elif target_encoding == "Transformer":
            i_layer = 0
            if input_len > max_seq_len:
                #####
                # for visualizing 1D LRP score
                #####
                # pad_lrp_layer = np.zeros(seq_len)
                # LRP_layer_len = LRP_per_layer[i_layer].shape[-1]
                # pad_lrp_layer[:LRP_layer_len] = LRP_per_layer[i_layer][0].sum(axis=0)[:LRP_layer_len]
                # lrp_to_vis.append(pad_lrp_layer)
                # base_seq = np.zeros((4, seq_len)) # for visualizing protein functional domains

                #####
                # for visualizing 2D LRP score
                #####
                pad_lrp_layer = np.zeros((LRP_per_layer[i_layer][0].shape[0], max_seq_len))
                LRP_layer_len = LRP_per_layer[i_layer].shape[-1]
                pad_lrp_layer[:, :LRP_layer_len] = LRP_per_layer[i_layer][0][:, :LRP_layer_len]
                lrp_to_vis += list(pad_lrp_layer)
                base_seq = np.zeros((4, max_seq_len)) # for visualizing protein functional domains
            else:
                #####
                # for visualizing 1D LRP score
                #####
                # print(LRP_per_layer[i_layer][0].shape)
                # lrp_to_vis.append(LRP_per_layer[i_layer][0].sum(axis=0)[:input_len].numpy())
                # base_seq = np.zeros((4, input_len))
                # seq_len = input_len  

                #####
                # for visualizing 2D LRP score
                #####

                lrp_to_vis += list(LRP_per_layer[i_layer][0][:, :input_len].numpy())
                base_seq = np.zeros((4, input_len)) 

        else:
            NotImplementedError

        # b = 10*((np.abs(lrp_to_vis)**3.0).mean()**(1.0/3)) # color scale for visualizing the LRP
        b = 0.25 * (np.array(lrp_to_vis).size**(1.0/3))*((np.abs(lrp_to_vis)**3.0).mean()**(1.0/3))

        if isinstance(cc, list):
            for segment in cc:
                start_idx = int(segment[0])-1
                end_idx = int(segment[1])-1
                base_seq[0, start_idx:end_idx] = -b
        if isinstance(dm, dict):
            for segments in dm.values():
                for segment in segments:
                    start_idx = int(segment[0])-1
                    end_idx = int(segment[1])-1
                    base_seq[1, start_idx:end_idx] = -0.5 * b
        if isinstance(zf, dict):
            for segments in zf.values():
                for segment in segments:
                    start_idx = int(segment[0])-1
                    end_idx = int(segment[1])-1
                    base_seq[2, start_idx:end_idx] = 0.5 * b
        if isinstance(rg, dict):
            for segments in rg.values():
                for segment in segments:
                    start_idx = int(segment[0])-1
                    end_idx = int(segment[1])-1
                    base_seq[3, start_idx:end_idx] = b


        lrp_to_vis += list(base_seq)

        lrp_to_vis = np.array(lrp_to_vis)
        heatmap = apply_heatmap(lrp_to_vis, 80, 5)

        ax = heatmap.axes
        ax.set_xticks(np.arange(0, input_len, step=1))
        ax.set_xticklabels(seq[:input_len])
        ax.set_xlabel('X-axis (Bottom)') 

        ax.set_yticks(np.arange(len(lrp_to_vis)-4, len(lrp_to_vis)))
        ax.set_yticklabels(['CC', 'DM', 'ZF', 'RG'])
        ax.set_ylabel('Y-axis')

        # Create a second axis that shares the same x-axis with the first  
        ax1 = ax.secondary_xaxis('top')
        # Set labels for the second axis (top)  

        ax1.set_xlabel('X-axis (Top)')  

        # Optionally set ticks for the second x-axis  
        ax1.set_xticks(np.arange(0, input_len, step=25))  # Specify desired ticks  
        ax1.set_xticklabels(np.arange(0, input_len, step=25))  # Optional custom tick labels  


        save_dir = './output/uniprotkb_trim_AND_reviewed_true_2024_12_04/figures/{}/{}'.format(cfg['test']['exp_id'], 'softmax')
        os.makedirs(save_dir, exist_ok=True)
        heatmap.figure.savefig('{}/LRP_out_{}_TC={}.pdf'.format(save_dir, id, target_class))

def visualize_att(model, df, target_encoding, inputs, max_seq_len=1024, bpe_encoding=1):
    IDs = df['ID']
    seqs = df['X']
    ccs = df['cc'] # coiled coil
    dms = df['dm'] # domain
    zfs = df['zf'] # zinc finger
    rgs = df['rg'] # region

    for id, input, cc, dm, zf, rg, seq in zip(IDs, inputs, ccs, dms, zfs, rgs, seqs):
        # input = 'MDGSGPFSCPICLEPLREPVTLPCGHNFCLACLGALWPHRSAGGTGGSGGPARCPLCQEPFPDGLQLRKNHTLSELLQLRQGSVPGPMSAPASGSTRGATPEPSAPSAPPPAPEPSAPCAPEQWPAGEEPVRCDACPEGAALPAALSCLSCLASFCSAHLAPHERSPALRGHRLVPPLRRLEESLCPRHLRPLERYCRVERVCLCEACATQDHRGHELVPLEQERALQEVEQSKVLSAAEDRMDELGAGIAQSRRTVALIKSAAVAERERVSQMFAEATATLQSFQNEVMGFIEEGEATMLGRSQGDLRRQEEQRSRLSKARHNLGQVPEADSVSFLQELLALRLALEEGCGPGPGPPRELSFTKSSQVVKAVRDTLISACASQWEQLRGLGSNEDGLQKLGSEDVESQDPDSTSLLESEAPRDYFLKFAYIVDLDSDTADKFLQLFGTKGVKRVLCPINYPESPTRFTHCEQVLGEGALDRGTYYWEVEIIEGWVSVGVMAEGFSPQEPYDRGRLGRNAHSCCLQWNGRGFSVWFCGLEAPLPHAFSPTVGVCLEYADHALAFYAVRDGKLSLLRRLKASRPRRSGALASPTDPFQSRLDSHFSGLFNHRLKPAFFLESVDAHLQIGPLKKSCITVLKRR'
        input_len = len(seq)
        # print(protein_2_embed(trans_protein(input))[0, :])
        # prep_input = torch.tensor(protein_2_embed(trans_protein(input))).double()


        if target_encoding == "CNN":
            prep_input = torch.tensor(protein_2_embed(trans_protein(input))).double()

        elif target_encoding == "ESM_embed":
            prep_input = torch.tensor(input) # 1 * 1024 * 1280
            prep_input = prep_input.unsqueeze(0).transpose(1, 2) # 1 * 1280 * 1024

        elif target_encoding == "Transformer":
            if bpe_encoding:
                prep_input = protein2emb_encoder(input, max_seq_len)
            else:                    
                prep_input = protein2emb_encoder_custom(input, max_seq_len)
            # prep_input = (torch.tensor(prep_input[0].reshape(1, -1)), torch.tensor(prep_input[1].reshape(1, -1))) # seq, mask
            prep_input = torch.stack([torch.tensor(i).unsqueeze(0) for i in prep_input], axis=1) # bs * 2 * seq_len
            output = model([prep_input])
            att_per_layer = [i.detach().numpy() for i in output[1][0]] # n_layer * bs * n_head * n_seq * n_seq
        else:
            NotImplementedError

        # Generate visualization(s)
        num_class = 2
        target_class = 1
        softmax = 1

        att_to_vis = []


        if target_encoding == "CNN":
            NotImplementedError
        
        elif target_encoding == "ESM_embed":
            NotImplementedError


        elif target_encoding == "Transformer":

            if input_len > max_seq_len:
                #####
                # for visualizing 1D LRP score
                #####
                # pad_lrp_layer = np.zeros(seq_len)
                # LRP_layer_len = LRP_per_layer[i_layer].shape[-1]
                # pad_lrp_layer[:LRP_layer_len] = LRP_per_layer[i_layer][0].sum(axis=0)[:LRP_layer_len]
                # lrp_to_vis.append(pad_lrp_layer)
                # base_seq = np.zeros((4, seq_len)) # for visualizing protein functional domains

                #####
                # for visualizing 2D LRP score
                #####
                att_to_vis += [att_layer_i[0].sum(axis=(0, 1)) for att_layer_i in att_per_layer]
                base_seq = np.zeros((4, max_seq_len)) # for visualizing protein functional domains
            else:
                #####
                # for visualizing 1D LRP score
                #####
                # print(LRP_per_layer[i_layer][0].shape)
                # lrp_to_vis.append(LRP_per_layer[i_layer][0].sum(axis=0)[:input_len].numpy())
                # base_seq = np.zeros((4, input_len))
                # seq_len = input_len  

                #####
                # for visualizing 2D LRP score
                #####
                att_to_vis += [att_layer_i[0].sum(axis=(0, 1))[:input_len] for att_layer_i in att_per_layer]
                base_seq = np.zeros((4, input_len)) 

        else:
            NotImplementedError

        # b = 10*((np.abs(lrp_to_vis)**3.0).mean()**(1.0/3)) # color scale for visualizing the LRP
        b = 0.25 * (np.array(att_to_vis).size**(1.0/3))*((np.abs(att_to_vis)**3.0).mean()**(1.0/3))

        if isinstance(cc, list):
            for segment in cc:
                start_idx = int(segment[0])-1
                end_idx = int(segment[1])-1
                base_seq[0, start_idx:end_idx] = -b
        if isinstance(dm, dict):
            for segments in dm.values():
                for segment in segments:
                    start_idx = int(segment[0])-1
                    end_idx = int(segment[1])-1
                    base_seq[1, start_idx:end_idx] = -0.5 * b
        if isinstance(zf, dict):
            for segments in zf.values():
                for segment in segments:
                    start_idx = int(segment[0])-1
                    end_idx = int(segment[1])-1
                    base_seq[2, start_idx:end_idx] = 0.5 * b
        if isinstance(rg, dict):
            for segments in rg.values():
                for segment in segments:
                    start_idx = int(segment[0])-1
                    end_idx = int(segment[1])-1
                    base_seq[3, start_idx:end_idx] = b


        att_to_vis += list(base_seq)

        att_to_vis = np.array(att_to_vis)
        heatmap = apply_heatmap(att_to_vis, 80, 5)

        ax = heatmap.axes
        ax.set_xticks(np.arange(0, input_len, step=1))
        ax.set_xticklabels(seq[:input_len])
        ax.set_xlabel('X-axis (Bottom)') 

        ax.set_yticks(np.arange(len(att_to_vis)-4, len(att_to_vis)))
        ax.set_yticklabels(['CC', 'DM', 'ZF', 'RG'])
        ax.set_ylabel('Y-axis')

        # Create a second axis that shares the same x-axis with the first  
        ax1 = ax.secondary_xaxis('top')
        # Set labels for the second axis (top)  

        ax1.set_xlabel('X-axis (Top)')  

        # Optionally set ticks for the second x-axis  
        ax1.set_xticks(np.arange(0, input_len, step=25))  # Specify desired ticks  
        ax1.set_xticklabels(np.arange(0, input_len, step=25))  # Optional custom tick labels  


        save_dir = './output/uniprotkb_trim_AND_reviewed_true_2024_12_04/figures/{}/{}'.format(cfg['test']['exp_id'], 'softmax')
        os.makedirs(save_dir, exist_ok=True)
        heatmap.figure.savefig('{}/att_out_{}_TC={}.pdf'.format(save_dir, id, target_class))

if __name__ == '__main__':
    ##########################################
    # for capsid protein lrp score calculation
    ##########################################

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
        df = pd.read_pickle("./data/uniprotkb_trim_AND_reviewed_true_2024_12_04_processed_full-v4.pkl") # trim data without esm embedding
        key_lst = ['Entry',	'Entry Name', 'X', 'Subclass', 'Predicted Score', 'lrp_score', 'lrp_segments', 'lrp_aa_segments', 'att_score', 'att_segments', 'att_aa_segments']
        origin_key_lst = ['Entry',	'Entry Name', 'X', 'Subclass']
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

        
    i_layer = 0
    num_class = 2
    target_class = 1
    normalize = 0
    softmax = 1
    offset_lst = [0, 3, 3, 10, 10, 21, 21] # based on conv layer kernel size and stride

    max_seq_len = cfg["backbone"]["deeppurpose"]["max_len_protein"]
    target_encoding = cfg['dataset']['protein_encoding']
    output_dir = "./output/uniprotkb_trim_AND_reviewed_true_2024_12_04/test/{}".format(cfg["test"]["exp_id"])

    os.makedirs(output_dir, exist_ok=True)
    lrp_score(model=model, i_layer=i_layer, num_class=num_class, target_class=target_class, df=df, 
              key_lst=key_lst, origin_key_lst=origin_key_lst, target_encoding=target_encoding,
              output_dir=output_dir,
              normalize=normalize,
              bpe_encoding=0)
    
    if target_encoding == "Transformer":
        att_score(model=model, i_layer=i_layer, target_class=target_class, df=df, 
                  key_lst=key_lst, origin_key_lst=origin_key_lst, target_encoding=target_encoding,
                  output_dir=output_dir,
                  normalize=normalize,
                  bpe_encoding=0)
    

