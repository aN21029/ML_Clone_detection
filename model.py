from datasets import load_dataset
import util

SLITHER_CLASS_TO_VULNERABILITY = {0: 'uninitialized-state', 1: 'constant-function-asm', 2: 'locked-ether',
                                  3: 'incorrect-shift', 4: 'divide-before-multiply', 5: 'unused-return',
                                  6: 'write-after-write', 7: 'reentrancy-no-eth', 8: 'unchecked-lowlevel',
                                  9: 'incorrect-equality', 10: 'weak-prng', 11: 'arbitrary-send',
                                  12: 'uninitialized-local', 13: 'reentrancy-eth', 14: 'shadowing-abstract',
                                  15: 'controlled-delegatecall', 16: 'unchecked-transfer', 17: 'erc20-interface',
                                  18: 'controlled-array-length', 19: 'tautology', 20: 'shadowing-state',
                                  21: 'tx-origin', 22: 'unprotected-upgrade', 23: 'suicidal',
                                  24: 'boolean-cst', 25: 'unchecked-send', 26: 'msg-value-loop',
                                  27: 'erc721-interface', 28: 'constant-function-state', 29: 'delegatecall-loop', 
                                  30: 'mapping-deletion', 31: 'reused-constructor', 32: 'uninitialized-storage',
                                  33: 'public-mappings-nested', 34: 'array-by-reference', 35: 'backdoor',
                                  36: 'rtlo', 37: 'name-reused', 38: 'safe'}

SLITHER_VULNERABLITIY_TO_LABEL = {"suicidal": "access-control", "arbitrary-send": "access-control", "unused-return": "unchecked-calls",
                                  "unchecked-lowlevel": "unchecked-calls", "erc20-interface": "ignore", "msg-value-loop": "double-spending",
                                  "controlled-array-length": "access-control", "locked-ether": "locked-ether", "shadowing-state": "ignore",
                                  "tx-origin": "access-control", "delegatecall-loop": "double-spending", "shadowing-abstract": "ignore",
                                  "constant-function-state": "other", "divide-before-multiply": "arithmetic", "array-by-reference": "other",
                                  "incorrect-equality": "other", "name-reused": "ignore", "incorrect-shift": "ignore",
                                  "rtlo": "other", "backdoor": "other", "write-after-write": "ignore", "weak-prng": "bad-randomness",
                                  "public-mappings-nested": "ignore", "controlled-delegatecall": "access-control", "reentrancy-no-eth": "reentrancy",
                                  "constant-function-asm": "other", "reused-constructor": "ignore", "reentrancy-eth": "reentrancy",
                                  "unprotected-upgrade": "access-control", "uninitialized-storage": "other", "tautology": "ignore",
                                  "unchecked-send": "locked-ether", "erc721-interface": "ignore", "uninitialized-local": "other",
                                  "unchecked-transfer": "unchecked-calls", "boolean-cst": "ignore", "mapping-deletion": "other",
                                  "uninitialized-state": "other", "safe": "safe"}

SLITHER_LABEL_MAPPINGS = {"access-control": 0, "arithmetic": 1, "other": 2,
                          "reentrancy": 3, "safe": 4, "unchecked-calls": 5,
                          "locked-ether": 6, "bad-randomness": 7, "double-spending": 8}

SLITHER_CLASSES = ["access-control", "arithmetic", "other", "reentrancy", "safe", "unchecked-calls",]

def get_class_label(class_num):
    class_name = SLITHER_VULNERABLITIY_TO_LABEL[SLITHER_CLASS_TO_VULNERABILITY[class_num]]
    # if label is ignore, we discard such a vulnerablity
    return SLITHER_LABEL_MAPPINGS[SLITHER_VULNERABLITIY_TO_LABEL[SLITHER_CLASS_TO_VULNERABILITY[class_num]]] if class_name != "ignore" else None

def get_class_labels(class_lst):
    # first initialize as set to make sure no duplicate classes, then convert to list to be able to input into analyzer
    return list({get_class_label(class_num) for class_num in class_lst if get_class_label(class_num)})


def data_collect():
    # Huggingface dataset for slither audited smart contracts
    # link: https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts#dataset-summary

    # Different formats:
    #   1. multilabel
    #   2. plain-text

    # Sizes: small(~10000), big(~100000), or all (~120000) smart contracts

    # Choice for this data collection is all-multilabel. multilabel for ease of classification
    # in model training, and all smart contracts from the database will be used to maximize results of model.

    # Dataset breakdown:
    # Breakdown of class distribution (see label mappings above) {access-control: 0, arithmetic: 20044, other: 32164, reentrancy: 34728, safe: 35130, unchecked-calls: 52080, locked-ether: 7768, bad-randomness: 3024, double-spending: 501}
    # Total number of contracts: 120608
    return load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel')["train"]

def data_collect2():
    # Huggingface dataset for slither audited smart contracts
    # link: https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts#dataset-summary

    # Different formats:
    #   1. multilabel
    #   2. plain-text

    # Sizes: small(~10000), big(~100000), or all (~120000) smart contracts

    # Choice for this data collection is all-multilabel. multilabel for ease of classification
    # in model training, and all smart contracts from the database will be used to maximize sample size for the model.
    return load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel')["train"]


def get_class_label(class_num):
    class_name = SLITHER_VULNERABLITIY_TO_LABEL[SLITHER_CLASS_TO_VULNERABILITY[class_num]]

    return SLITHER_LABEL_MAPPINGS[SLITHER_VULNERABLITIY_TO_LABEL[SLITHER_CLASS_TO_VULNERABILITY[class_num]]] if class_name != "ignore" else None

def get_class_labels(class_lst):
    # first initialize as set to make sure no duplicate classes, then convert to list to be able to input into analyzer
    return list({get_class_label(class_num) for class_num in class_lst if get_class_label(class_num)})

if __name__ == "__main__":
    # data = data_collect()
    # print(total)

    data = data_collect()

    cl = util.CodeLevelAnalyzer(data)
    # print(cl.test_embeddings)    
    for i in range(10):
        print(cl.train_labels)
        print(cl._predict_test_set_classes(0.9 + (i/100)))
    # print(cl.model_statistics())
