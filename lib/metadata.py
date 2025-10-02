# -*- coding: utf-8 -*-
# Author: ImDrug Team
# License: MIT 

"""This file contains all metadata of datasets in DrugLT.
Attributes:
    adme_dataset_names (list): all adme dataset names
    admet_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    admet_splits (dict): a dictionary with key the dataset name and value the recommended split
    catalyst_dataset_names (list): all catalyst dataset names
    category_names (dict): mapping from ML problem (1st tier) to all tasks
    crisproutcome_dataset_names (list): all crispr outcome dataset names
    dataset_list (list): total list of dataset names in ImDrug
    dataset_names (dict): mapping from task name to list of dataset names
    ddi_dataset_names (list): all ddi dataset names
    develop_dataset_names (list): all develop dataset names
    distribution_oracles (list): all distribution learning oracles, i.e. molecule evaluators
    download_oracle_names (list): oracle names that require downloading predictors
    drugres_dataset_names (list): all drugres dataset names
    drugsyn_dataset_names (list): all drugsyn dataset names
    drugsyn_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    drugsyn_splits (dict):  a dictionary with key the dataset name and value the recommended split
    dti_dataset_names (list): all dti dataset names
    dti_dg_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    dti_dg_splits (dict):  a dictionary with key the dataset name and value the recommended split
    evaluator_name (list): list of evaluator names
    forwardsyn_dataset_names (list): all reaction dataset names
    generation_datasets (list): all generation dataset names
    meta_oracle_name (list): list of all meta oracle names
    molgenpaired_dataset_names (list): all molgenpaired dataset names
    mti_dataset_names (list): all mti dataset names
    name2stats (dict): mapping from dataset names to statistics
    name2type (dict): mapping from dataset names to downloaded file format
    oracle2id (dict): mapping from oracle names to dataverse id
    oracle2type (dict): mapping from oracle names to downloaded file format
    receptor2id (dict): mapping from receptor id to dataverse id 
    oracle_names (list): list of all oracle names
    paired_dataset_names (list): all paired dataset names
    ppi_dataset_names (list): all ppi dataset names
    property_names (list): a list of oracles that correspond to some molecular properties
    qm_dataset_names (list): all qm dataset names
    retrosyn_dataset_names (list): all retrosyn dataset names
    single_molecule_dataset_names (list): all molgen dataset names
    synthetic_oracle_name (list): all oracle names for synthesis
    test_multi_pred_dataset_names (list): test multi pred task name
    test_single_pred_dataset_names (list): test single pred task name
    toxicity_dataset_names (list): all toxicity dataset names
    trivial_oracle_names (list): a list of oracle names for trivial oracles
    yield_dataset_names (list): all yield dataset names
"""
####################################
# test cases
test_single_pred_dataset_names = ['test_single_pred']
test_multi_pred_dataset_names = ['test_multi_pred']

# single_pred prediction

toxicity_dataset_names = ['tox21']

adme_dataset_names = ['bbb_martins']

bioact_dataset_names = ['hiv']

qm_dataset_names = ['qm9']

transpos_dataset_names = [
'debug',
'PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k',
'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k',
'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k_nodup',
'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k_noseqdup',
'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k_noseqdup_minus_uniprot_9606_2023_10_12',
'XDM_partial',
'esm2_t33_650M_UR50D-maxp1024-nodup',
'esm2_t33_650M_UR50D-maxp1024-noseqdup',
'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k-maxp545-esm2_t6_8M_UR50D',
'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k-maxp1024-esm2_t6_8M_UR50D',
'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k-maxp545-esm2_t33_650M_UR50D',
'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k-maxp1024-esm2_t33_650M_UR50D',
'tn_droped_decoy_1m',
'protein_tp_2',
'protein_tp',
'protein_capsid_homo',
'protein_10090_16-4096',
'protein_9606_16-4096',
'protein_10090',
'protein_9606',
'protein_16-4096',
'protein_line',
'protein_class_1and3',
'protein_class_1and3_with*',
'protein_rt',
'protein_rt_gag',
'protein_line_2',
'protein_rt_2',
'protein_rt_gag_2',
'protein_retrotransposon',
'zf',
'Capsid-T_number_v1',
'0.98-2023-06-20-22_52-query_downloaded_VF202306_decoy1m',
'0.99-2023-06-20-23_30-query_downloaded_VF202306_decoy1m',
'0.99-2023-08-30-16_02-query_Merged_PF_label012_unique',
'99_merged_pf_unique',
'99_merged_pf_unique_260k_neg_noseqdup',
'uniprot_9606_2023_10_12',
'uniprot_9606_2023_10_12_unique',
'uniprot_9606_2023_10_12_hmmscan',
'uniprotkb_trim_AND_reviewed_true_2024_12_04',
'uniprotkb_trim_AND_reviewed_true_2024_12_04-esm2_t33_650M_UR50D-maxp1024',
'PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1',
'PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup',
'PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup_minus_uniprot_9606_2023_10_12_nonhuman',
'literature_test',
'TRIM25_variants',
'TRIM25-variants-v1',
'TRIM25-variants-v2']


####################################
# multi_pred prediction

dti_dataset_names = [
                     'sbap',
                     'sbap_reg']

ddi_dataset_names = ['drugbank', 'uspto_50k', 'uspto_1k_TPL', 'uspto_500_MT', 'uspto_catalyst']

ppi_dataset_names = []

yield_dataset_names = ['uspto_yields', 'uspto_500_MT']

catalyst_dataset_names = ['uspto_catalyst']

reacttype_dataset_names = ['uspto_50k', 'uspto_1k_TPL', 'uspto_500_MT']

####################################
# generation

retrosyn_dataset_names = ['uspto50k', 'uspto']

forwardsyn_dataset_names = ['uspto']

single_molecule_dataset_names = ['zinc', 'moses', 'chembl', 'chembl_v29']

paired_dataset_names = ['uspto50k', 'uspto']

####################################
# oracles

evaluator_name = ['roc-auc', 'f1', 'pr-auc', 'precision', 'recall', \
				  'accuracy', 'balanced_accuracy', 'mse', 'rmse', 'mae', 'r2', 'micro-f1', 'macro-f1', \
				  'weighted-f1', 'balanced-f1', 'kappa', 'avg-roc-auc', 'rp@k', 'pr@k', 'pcc', 'spearman']

####################################

category_names = {'single_pred': ["Tox",
									"ADME",
									"BioAct",
									"QM",
									"Yields",],
				'multi_pred': ["DTI",
								"PPI",
								"DDI",
								"Catalyst"],
				'generation': ["RetroSyn",
								"Reaction",
								"MolGen"
								]
				}
input_names = {'multi_pred': {"DTI": ['drug', 'protein'],
								"PPI": ['protein', 'protein'],
								"DDI": ['drug', 'drug'],
								"Catalyst": ['drug', 'drug'],
							}	
				}


def get_task2category():
	task2category = {}
	for i, j in category_names.items():
		for x in j:
			task2category[x] = i
	return task2category

dataset_names = {"Tox": toxicity_dataset_names,
				"ADME": adme_dataset_names, 
				'BioAct': bioact_dataset_names,
				"DTI": dti_dataset_names, 
				"PPI": ppi_dataset_names, 
				"DDI": ddi_dataset_names,
				"QM": qm_dataset_names,
				"Yields": yield_dataset_names, 
				"ReactType": reacttype_dataset_names, 
				"Catalyst": catalyst_dataset_names, 
				"test_single_pred": test_single_pred_dataset_names,
				"test_multi_pred": test_multi_pred_dataset_names,
				"Transposition": transpos_dataset_names, # WARNING: cannot be published
				}

dataset_list = []
for i in dataset_names.keys():
    dataset_list = dataset_list + [i.lower() for i in dataset_names[i]]

name2type = {'toxcast': 'tab',
 'tox21': 'tab',
 'bbb_martins': 'tab',
 'hiv': 'tab',
 'drugbank': 'csv',
 'uspto50k': 'tab',
 'qm9': 'csv',
 'uspto_50k': 'csv',
 'uspto_1k_TPL': 'csv',
 'uspto_500_MT': 'csv',
 'uspto_yields': 'csv',
 'uspto_catalyst': 'csv',
 'test_single_pred': 'tab',
 'test_multi_pred': 'tab',
 'protein_16-4096': 'csv',
 'protein_line': 'csv',
 'protein_rt_gag': 'csv',
 'protein_rt': 'csv',
 'protein_line_2': 'csv',
 'protein_rt_gag_2': 'csv',
 'protein_rt_2': 'csv',
 'protein_class_1and3':'csv',
 'protein_class_1and3_with*':'csv',
 'protein_retrotransposon':'csv',
 'protein_9606':'csv',
 'protein_10090':'csv',
 'protein_10090_16-4096':'csv',
 'protein_9606_16-4096':'csv',
 'protein_capsid_homo':'csv',
 'protein_tp':'csv',
 'protein_tp_2':'csv',
 'sbap': 'csv',
 'sbap_reg': 'csv',
 'PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k':'csv',
 'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k':'csv',
 'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k_nodup':'csv',
 'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k_noseqdup':'csv',
 'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k_noseqdup_minus_uniprot_9606_2023_10_12':'csv',
 'esm2_t33_650M_UR50D-maxp1024-nodup': 'csv',
 'esm2_t33_650M_UR50D-maxp1024-noseqdup': 'csv',
 'XDM_partial': 'csv',
 'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k-maxp545-esm2_t6_8M_UR50D':'csv',
 'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k-maxp1024-esm2_t6_8M_UR50D':'csv',
 'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k-maxp545-esm2_t33_650M_UR50D':'csv',
 'PS_revised_labeled_coat+capsid_20221009_retrovirus_list_decoy_260k-maxp1024-esm2_t33_650M_UR50D':'csv',
 'tn_droped_decoy_1m':'csv',
 'zf':'csv',
 'Capsid-T_number_v1':'csv',
 '0.98-2023-06-20-22_52-query_downloaded_VF202306_decoy1m':'csv',
 '0.99-2023-06-20-23_30-query_downloaded_VF202306_decoy1m':'csv',
 '0.99-2023-08-30-16_02-query_Merged_PF_label012_unique':'csv',
 '99_merged_pf_unique':'csv',
 '99_merged_pf_unique_260k_neg_noseqdup':'csv',
 'debug': 'csv',
 'uniprot_9606_2023_10_12': 'csv',
 'uniprot_9606_2023_10_12_unique': 'csv',
 'uniprot_9606_2023_10_12_hmmscan': 'csv',
 'uniprotkb_trim_AND_reviewed_true_2024_12_04': 'csv',
 'uniprotkb_trim_AND_reviewed_true_2024_12_04-esm2_t33_650M_UR50D-maxp1024': 'csv',
 'PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1': 'csv',
 'PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup': 'csv',
 'PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup_minus_uniprot_9606_2023_10_12_nonhuman': 'csv',
 'literature_test': 'csv',
 'TRIM25_variants': 'csv',
 'TRIM25-variants-v1': 'csv',
 'TRIM25-variants-v2': 'csv',}

name2stats = {
	'bbb_martins': 1975,
	'tox21': 7831,
	'hiv': 41127,
	'qm9': 133885,
	'sbap': 32140,
	'uspto_yields': 853638,
	'uspto-50k': 50016,
	'uspto-500-MT': 143535,
	'uspto-1k-TPL': 445115,
	'drugbank': 191808,
	'uspto_catalyst': 721799,
}

name2imratio = {
	'tox21': 22.51, 
    'bbb_martins': 3.24,
	'hiv': 27.50,
	'qm': 133883, 
	'sbap': 36.77,
	'uspto_50k': 65.78,
	'uspto_1k_TPL': 110.86,
	'uspto_500_MT': 285.06,
	'uspto_catalyst': 3975.86,
	'uspto_yields': 7.59,
	'drugbank': 10124.67
}

metrics = {'Imbalanced Learning': ['balanced_accuracy', 'balanced-f1', 'roc-auc', 'recall', 'precision', 'accuracy', 'f1'],
		   'LT Classification': ['balanced_accuracy', 'balanced-f1', 'roc-auc'],
		   'Open LT': ['balanced_accuracy', 'balanced-f1', 'roc-auc'],
		   'LT Regression': ['mse', 'mae']}
