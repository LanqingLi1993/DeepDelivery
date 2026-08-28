## DeepDelivery: AI-driven discovery and engineering of human-derived protein scaffolds for mRNA delivery 

# Reproducing environment

    GPU: Tesla V100S-PCIE-32GB
    CPU: Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
    NVIDIA-SMI: 550.54.14
    CUDA Version: 12.4

Conda environment can be installed from environment.yml.

# Model inference on human proteome data

The model 1 presented in the paper is a relatively older version originally used for initial coarse screening, since then we have iterated many versions. We are providing a latest version which should predict 1333 positive samples with AA length &le; 512 from our human proteome dataset. The following two-stage screening pipeline should still identify all the 15 sequences used for experimental validation in our paper.

## Model 1 screening
```bash
    python script/test.py --config ./lib/config/model1.json
```

## Model 2 screening
```bash
    python script/test.py --config ./lib/config/model2.json
```

Outputs are stored in "./output/uniprot_9606_2023_10_12_unique/test" by default.

# Computing LRP score for TRIM family proteins
```bash
    python xai/lrp.py  --config ./lib/config/model1.json
```

Outputs are stored in "./output/uniprotkb_trim_AND_reviewed_true_2024_12_04/test" by default.