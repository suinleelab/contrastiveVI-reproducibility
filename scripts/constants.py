"""Global constants for project scripts."""

DATASET_LIST = [
    "zheng_2017",
    "haber_2017",
    "blish_2020",
    "xiang_2020",
    "fasolino_2021",
    "mcfarland_2020",
    "grubman_2019",
    "norman_2019",
    "cain_2020",
    "cain_2020_excluding_cog_only",
    "mcginnis_2019",
    "srivatsan_2019",
    "papalexi_2021",
]
DEFAULT_DATA_PATH = "./data"  # Modify to preferred data storage path.
DEFAULT_RESULTS_PATH = "./results"  # Modify to preferred result storage path.
DEFAULT_SEEDS = [123, 42, 789, 46, 999]

PROTEIN_EXPRESSION_KEY = "protein_expression"

NORMALIZATION_LIST = ["tc", "tmm", "scran", "basics"]
METHODS_WITHOUT_LIB_NORMALIZATION = ["PCPCA", "cPCA", "cVAE"]

DATASET_SPLIT_LOOKUP = {
    "zheng_2017": {
        "split_key": "condition",
        "background_value": "healthy",
        "label_key": "condition",
    },
    "haber_2017": {
        "split_key": "condition",
        "background_value": "Control",
        "label_key": "condition",
    },
    "fasolino_2021": {
        "split_key": "disease_state",
        "background_value": "Control",
        "label_key": "disease_state",
    },
    "mcfarland_2020": {
        "split_key": "condition",
        "background_value": "DMSO",
        "label_key": "TP53_mutation_status",
    },
    "norman_2019": {
        "split_key": "gene_program",
        "background_value": "Ctrl",
        "label_key": "gene_program",
    },
    "grubman_2019": {"split_key": "batchCond", "background_value": "ct"},
    "cain_2020": {
        "split_key": "combined_diagnosis",
        "background_value": "None",
        "label_key": "combined_diagnosis",
    },
    "cain_2020_excluding_cog_only": {
        "split_key": "combined_diagnosis",
        "background_value": "None",
        "label_key": "combined_diagnosis",
    },
    "mcginnis_2019": {
        "split_key": "TumorStage",
        "background_value": "WT",  # WT (= Wild type) means no cancer here
        "label_key": "TumorStage",
    },
    "srivatsan_2019": {
        "split_key": "drug",
        "background_value": "Vehicle",
        "label_key": "drug",
    },
    "papalexi_2021": {
        "split_key": "crispr",
        "background_value": "NT",
    },
}
