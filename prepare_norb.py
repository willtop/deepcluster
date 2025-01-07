"""
Using torchvision, download celebA dataset into a local folder
Then arrange the folder by splits, into a pseudo class
such that it's compatible with ImageFolder()
"""

import os
import numpy as np
import datetime
from tqdm import tqdm
import tensorflow_datasets as tfds
from PIL import Image


NORB_DIR = "datasets/norb"

if __name__ == "__main__":
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), NORB_DIR)
    os.makedirs(img_dir, exist_ok=True)
    # load the NORB using tensorflow dataset
    # training set
    ds_tf_train = tfds.as_numpy(
                    tfds.load('smallnorb', 
                              data_dir=img_dir,
                              split='train', 
                              shuffle_files=False))
    ds_tf_test = tfds.as_numpy(
                    tfds.load('smallnorb',
                              data_dir=img_dir,
                              split='test',
                              shuffle_files=False))
    n_train_imgs, n_test_imgs = len(ds_tf_train), len(ds_tf_test)
    assert n_train_imgs == n_test_imgs == 24_300
    
    # arrange the images by split
    # since this is to be used for unsupervised deepcluster training
    # just have a pseudo class "cls1" created
    os.makedirs(os.path.join(img_dir, 'train', 'cls1'))
    os.makedirs(os.path.join(img_dir, 'test', 'cls1'))

    # save the images into corresponding split folder
    print("Saving training NORB images...")
    for i, sample in enumerate(tqdm(ds_tf_train)):
        img_raw = sample['image']
        # tile one channel to three channels here
        # since otherwise the Image.fromarray() reports error, understandably
        img_raw = np.tile(img_raw, (1,1,3))
        img = Image.fromarray(img_raw)
        img_filename = f'{i:05d}.jpg'
        img_path = os.path.join(img_dir, 'train', 'cls1', img_filename)
        assert not os.path.isfile(img_path)
        img.save(img_path)
    print("Finished saving to the training data dir: ", os.path.join(img_dir, 'train'))
    
    
    print("Saving training NORB images...")
    for i, sample in enumerate(tqdm(ds_tf_test)):
        img_raw = sample['image']
        # tile one channel to three channels here
        # since otherwise the Image.fromarray() reports error, understandably
        img_raw = np.tile(img_raw, (1,1,3))
        img = Image.fromarray(img_raw)
        img_filename = f'{i:05d}.jpg'
        img_path = os.path.join(img_dir, 'test', 'cls1', img_filename)
        assert not os.path.isfile(img_path)
        img.save(img_path)
    print("Finished saving to the testing data dir: ", os.path.join(img_dir, 'test'))

    print("Script finished!")