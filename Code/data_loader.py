import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

def load_video_data(dataset_dir, video_list, subset, img_height=240, img_width=240, img_channels=3):
    """
    Loads and preprocesses video data from the specified dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory.
        video_list (list): List of video names to load.
        subset (str): Subset name ('train' or 'test').
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
    Defines train and test video sets.

    Args:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        dict: Dictionary with 'train' and 'test' video lists.
    """
    video_sets = {
        "train": [
            '360___Video_of_a_Compartment_Fire_Burner_Shakedown_Video_Download',
            '360___Video_of_a_Fire_involving_Parallel_Privacy_Fences_on_August_1,_2019_Video_Download',
            'nofire18',
            '360___Video_of_a_Kitchen_Fire_Video_Download',
            '360___Video_of_a_Replica_Museum_Collection_Storage_Room_Fire_Video_Download',
            'nofire1',
            'flame1', 'flame2', 'flame3',
            'nofire2',
            'flame4', 'flame5', 'FireVid20',
            'nofire3',
            'FireVid21', 'FireVid22', 'FireVid23',
            'nofire4',
            'FireVid24', 'posVideo1', 'posVideo2',
            'nofire5',
            'posVideo3', 'posVideo4', 'posVideo5',
            'nofire6',
            'posVideo6', 'posVideo7', 'posVideo8',
            'nofire7',
            'posVideo9', 'posVideo10', 'posVideo11',
            'nofire8',
            'indoor_night_20m_heptane_CCD_001', 'indoor_night_20m_heptane_CCD_002',
            'nofire9',
            'outdoor_daytime_10m_gasoline_CCD_001', 'outdoor_daytime_10m_heptane_CCD_001',
            'nofire10',
            'outdoor_daytime_20m_gasoline_CCD_001', 'outdoor_daytime_20m_heptane_CCD_001',
            'nofire11',
            'outdoor_daytime_30m_gasoline_CCD_001', 'outdoor_daytime_30m_heptane_CCD_001',
            'nofire12',
            'outdoor_night_10m_gasoline_CCD_001', 'outdoor_night_10m_gasoline_CCD_002',
            'nofire13',
            'outdoor_night_10m_heptane_CCD_001', 'outdoor_night_20m_gasoline_CCD_001',
            'nofire14',
            'outdoor_night_20m_heptane_CCD_001', 'outdoor_night_20m_heptane_CCD_002',
            'nofire15',
            'outdoor_night_30m_gasoline_CCD_001', 'outdoor_night_30m_heptane_CCD_001',
            'nofire16',
            'outdoor_night_10m_heptane_CCD_002', 'rescuer_001', 'rescuer_002',
            'nofire17',
            'rescuer_004', 'rescuer_003',
            'nofire19',
        ],
        "test": ['Chino']
    }
    return video_sets