import numpy as np
import trimesh
import random
import os
from collections import Counter

def load_off_to_numpy(file_path):
    mesh = trimesh.load_mesh(file_path)
    if mesh.is_empty:
        print(f"File {file_path} contains no data.")
        return None
    vertices = np.asarray(mesh.vertices)
    return vertices

def load_all_off_in_folder(folder_path):
    data = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".off"):
            arr = load_off_to_numpy(os.path.join(folder_path, fname))
            if arr is not None:
                data.append(arr)
    return data

def shuffle_data(data, labels):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx], labels[idx]

def get_point_cloud_sizes(train_data, test_data):
    train_sizes = [data.shape[0] for data in train_data]
    test_sizes = [data.shape[0] for data in test_data]

    min_train_size = min(train_sizes) if len(train_sizes) > 0 else None
    max_train_size = max(train_sizes) if len(train_sizes) > 0 else None

    min_test_size = min(test_sizes) if len(test_sizes) > 0 else None
    max_test_size = max(test_sizes) if len(test_sizes) > 0 else None

    return min_train_size, max_train_size, min_test_size, max_test_size


def count_clouds_by_size(train_data, test_data):
    train_sizes = [data.shape[0] for data in train_data]
    test_sizes = [data.shape[0] for data in test_data]

    small, medium, large = 0, 0, 0

    for size in train_sizes + test_sizes:
        if size < 300:
            small += 1
        elif 300 <= size <= 5000:
            medium += 1
        else:
            large += 1

    return small, medium, large


def farthest_point_sampling(points, n_samples):
    N = points.shape[0]
    selected_points = np.zeros((n_samples, points.shape[1]))
    selected_idx = np.random.randint(0, N)  
    selected_points[0] = points[selected_idx]
    
    dist = np.linalg.norm(points - selected_points[0], axis=1)  
    
    for i in range(1, n_samples):
        dist = np.minimum(dist, np.linalg.norm(points - selected_points[i-1], axis=1))
        selected_idx = np.argmax(dist)
        selected_points[i] = points[selected_idx]
        
    return selected_points

def oversample_with_noise(cloud, target_size, noise_scale=0.01):
    num_points = cloud.shape[0]
    if num_points >= target_size:
        return cloud
    deficit = target_size - num_points
    sampled_points = cloud[np.random.choice(num_points, deficit, replace=True)]
    noise = np.random.normal(scale=noise_scale, size=sampled_points.shape)
    oversampled_points = sampled_points + noise
    processed_cloud = np.vstack((cloud, oversampled_points))
    
    return processed_cloud

def process_point_clouds(data, target_size):
    processed_data = []
    noise_scale=0.01
    for cloud in data:
        num_points = cloud.shape[0]
        if num_points <= target_size:
            #processed_data.append(cloud)
            processed_cloud = oversample_with_noise(cloud, target_size, noise_scale)
        else:
            processed_cloud = farthest_point_sampling(cloud, target_size)
        processed_data.append(processed_cloud)
    return processed_data
