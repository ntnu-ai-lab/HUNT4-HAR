import os
def getDataset(dataset):
    """
    Returns training dataset path and dataset name
    :return: dataset path, dataset name
    """
    __dataset = os.path.dirname(os.path.abspath(__file__))
    __dataset = __dataset.split('Recurrent_ANN', 1)[0]
    __dataset = __dataset + "/DATA/" + dataset

    # Build dataset name
    if "Downsampled" in __dataset:
        __dataset_name = __dataset.split('Downsampled-data/', 1)[1]
    else:
        __dataset_name = __dataset.split('DATA/', 1)[1]

    if "/" in __dataset_name:
        __dataset_name = __dataset_name.replace("/", "-")
    return __dataset, __dataset_name

def getNewLabels():
    """
    Returns relabeling dict for doing relabeling of data
    :return: relabel dict
    """
    # Dict for relabling data
    __relabel_dict = {
        3: 1,  # Shuffling -> Walking
        4: 1,  # Stairs -> Walking
        5: 1,  # Stairs -> Walking
        10: 6,  # Bending -> Standing
        11: 6,  # Picking -> Standing
        12: 9,  # Undefined -> trainsition
        14: 13,  # Cycling (stand) -> Cycling (sit)
        16: 2,  # Vigorous activity -> Running
        17: 1,  # Non-vigorous activity -> Walking
        18: 19,  # Transport (sitting) -> Commute (standing)
        20: 8,  # Lying
        21: 8,  # Lying
        22: 8,  # Lying
        23: 8  # Lying
    }
    return __relabel_dict
