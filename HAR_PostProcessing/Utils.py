def convert_string_labels_to_numbers(label_list):
    label_to_number_dict = {
        "walking": 1,
        "running": 2,
        "shuffling": 3,
        "stairs (ascending)": 4,
        "stairs (descending)": 5,
        "standing": 6,
        "sitting": 7,
        "lying": 8,
        "transition": 9,
        "bending": 10,
        "picking": 11,
        "undefined": 12,
        "cycling (sit)": 13,
        "cycling (stand)": 14,
        "heel drop": 15,
        "vigorous activity": 16,
        "non-vigorous activity": 17,
        "Car": 18  # TODO: Check with HAR group to see if this labeling of Car is all right.
    }
    return [label_to_number_dict[label] for label in label_list]
