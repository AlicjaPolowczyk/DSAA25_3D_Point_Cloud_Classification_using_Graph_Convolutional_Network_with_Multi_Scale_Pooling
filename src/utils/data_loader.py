import numpy as np
from src.utils.data_utils import (
    load_all_off_in_folder,
    shuffle_data,
    process_point_clouds,
    get_point_cloud_sizes,
    count_clouds_by_size,
)

def load_dataset(cfg):
    train_data, test_data = [], []
    train_labels, test_labels = [], []

    for idx, (class_name, paths) in enumerate(cfg.classes.items()):
        train_class = load_all_off_in_folder(paths.train)
        test_class = load_all_off_in_folder(paths.test)

        train_data += train_class
        test_data += test_class
        train_labels += [idx] * len(train_class)
        test_labels += [idx] * len(test_class)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train_data, train_labels = shuffle_data(np.array(train_data, dtype=object), train_labels)
    test_data, test_labels = shuffle_data(np.array(test_data, dtype=object), test_labels)

    min_train_size, max_train_size, min_test_size, max_test_size = get_point_cloud_sizes(train_data, test_data)
    print(f"Train sizes: min={min_train_size}, max={max_train_size}")
    print(f"Test sizes: min={min_test_size}, max={max_test_size}")

    small, medium, large = count_clouds_by_size(train_data, test_data)
    print(f"Point clouds < 300 points: {small}")
    print(f"Point clouds 300–5000 points: {medium}")
    print(f"Point clouds > 5000 points: {large}")

    train_data = process_point_clouds(train_data, target_size=cfg.target_size)
    test_data = process_point_clouds(test_data, target_size=cfg.target_size)

    return train_data, test_data, train_labels, test_labels
