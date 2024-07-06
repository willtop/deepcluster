"""
Using torchvision, download celebA dataset into a local folder
Then arrange the folder by splits, into a pseudo class
such that it's compatible with ImageFolder()
"""

import os
import numpy as np
import datetime
from tqdm import trange
from PIL import Image


MPI3D_DIR = "mpi3d_dataset"

if __name__ == "__main__":
    # load the MPI3D dataset from the downloaded npz file
    datafile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 MPI3D_DIR, 
                                 "real3d_complicated_shapes_ordered.npz")
    print(f"Loading mpi3d data from {datafile_path}...")
    start_time = datetime.datetime.now().replace(microsecond=0)
    mpi3d_data = np.load(datafile_path)['images']
    n_imgs = mpi3d_data.shape[0]
    assert n_imgs == 460800
    end_time = datetime.datetime.now().replace(microsecond=0)
    print(f"[MPI3D] data loaded, took time: {end_time-start_time}.")
    
    
    # arrange the images by split
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           MPI3D_DIR)
    # since this is to be used for unsupervised deepcluster training
    # just have a pseudo class "cls1" created
    os.makedirs(os.path.join(img_dir, 'train', 'cls1'))
    os.makedirs(os.path.join(img_dir, 'test', 'cls1'))

    # save the images into corresponding split folder
    print("Saving training MPI3D images...")
    for i in trange(400000):
        img = Image.fromarray(mpi3d_data[i])
        img_filename = f'{i+1:06d}.jpg'
        img_path = os.path.join(img_dir, 'train', 'cls1', img_filename)
        assert not os.path.isfile(img_path)
        img.save(img_path)
    
    print("Saving testing MPI3D images...")
    for i in trange(400000, 460800):
        img = Image.fromarray(mpi3d_data[i])
        img_filename = f'{i+1:06d}.jpg'
        img_path = os.path.join(img_dir, 'test', 'cls1', img_filename)
        assert not os.path.isfile(img_path)
        img.save(img_path)

    print("Training data dir: ", os.path.join(img_dir, 'train'))
    print("Testing data dir: ", os.path.join(img_dir, 'test'))

    print("Script finished!")