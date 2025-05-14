import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

def load_video_data(dataset_dir, video_list, subset, img_height=288, img_width=288, img_channels=3):
    """
    Loads and preprocesses video data from the specified dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory.
        video_list (list): List of video names to load.
        subset (str): Subset name ('test').
        img_height (int): Target image height.
        img_width (int): Target image width.
        img_channels (int): Number of image channels.

    Returns:
        tuple: (X, Y) where X is a list of normalized image arrays and Y is a list of mask arrays.
    """
    X, Y = [], []
    for video in video_list:
        image_dir = os.path.join(dataset_dir, subset, video, 'image')
        label_dir = os.path.join(dataset_dir, subset, video, 'label')

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        video_X, video_Y = [], []
        for id_ in tqdm(image_files, desc=f'Processing {subset}/{video}'):
            img_path = os.path.join(image_dir, id_)
            img = imread(img_path)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
            video_X.append(img)

            mask_file = id_.replace('.png', '_label.png')
            mask_path = os.path.join(label_dir, mask_file)
            if os.path.exists(mask_path):
                mask = imread(mask_path)
                if len(mask.shape) == 3:
                    mask = rgb2gray(mask)
                mask = resize(mask, (img_height, img_width), mode='constant', preserve_range=True)
                mask = np.expand_dims(mask, axis=-1)
                video_Y.append(mask)
            else:
                print(f"⚠ Warning: Mask file {mask_file} does not exist!")

        X.append(np.array(video_X, dtype=np.uint8) / 255.0)
        Y.append(np.array(video_Y, dtype=bool))
    return X, Y

def get_dataset_splits(dataset_dir):
    """
    Defines test video set.

    Args:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        dict: Dictionary with 'test' video list.
    """
    video_sets = {
        # "train": [
        #     '360___Video_of_a_Compartment_Fire_Burner_Shakedown_Video_Download',
        #     '360___Video_of_a_Fire_involving_Parallel_Privacy_Fences_on_August_1,_2019_Video_Download',
        #     'nofire1',
        #     'fire1', 'fire2'
        # ],  # Add this line if wanting to train
        "test": ['Chino']
    }
    return video_sets
