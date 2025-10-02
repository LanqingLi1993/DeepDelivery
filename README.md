## DeepDelivery: AI-driven discovery and engineering of human endogenous nanocage proteins for mRNA delivery

# Reproducing environment

    GPU: Tesla V100S-PCIE-32GB
    CPU: Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
    NVIDIA-SMI: 550.54.14
    CUDA Version: 12.4

Conda environment can be installed from environment.yml.

# Model inference on human proteome data

    python script/test.py --config ./lib/config/CNN.json

Outputs are stored in "./output/uniprot_9606_2023_10_12_unique/test" by default.

# Computing LRP score for TRIM family proteins

    python xai/lrp.py  --config ./lib/config/CNN.json

Outputs are stored in "./output/uniprotkb_trim_AND_reviewed_true_2024_12_04/test" by default.