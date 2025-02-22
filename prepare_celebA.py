"""
Using torchvision, download celebA dataset into a local folder
Then arrange the folder by splits, into a pseudo class
such that it's compatible with ImageFolder()
"""

import os
from torchvision.datasets.celeba import CelebA

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

if __name__ == "__main__":
    # load the CelebA using torchvision into the folder "celeba_dataset"
    # no need to specify the label type or transformation here
    _ = CelebA(DATA_DIR, 
                split="all",
                download=True)
    
    # arrange the images by split
    orig_img_dir = os.path.join(DATA_DIR,
                           "celeba",
                           "img_align_celeba")
    # since this is to be used for unsupervised deepcluster training
    # just have a pseudo class "cls1" created
    os.makedirs(os.path.join(DATA_DIR, "celeba", 'train', 'cls1'))
    os.makedirs(os.path.join(DATA_DIR, "celeba", 'test', 'cls1'))

    # move the images into corresponding split folder
    print("Moving training CelebA images...")
    for i in range(1, 162771):
        img_filename = f'{i:06d}.jpg'
        old_img_loc = os.path.join(orig_img_dir, img_filename)
        new_img_loc = os.path.join(DATA_DIR, "celeba", 'train', 'cls1', img_filename)
        assert os.path.isfile(old_img_loc)
        assert not os.path.isfile(new_img_loc)
        os.rename(old_img_loc, new_img_loc)
    
    print("Moving testing CelebA images...")
    for i in range(162771, 202600):
        img_filename = f'{i:06d}.jpg'
        old_img_loc = os.path.join(orig_img_dir, img_filename)
        new_img_loc = os.path.join(DATA_DIR, "celeba", 'test', 'cls1', img_filename)
        assert os.path.isfile(old_img_loc)
        assert not os.path.isfile(new_img_loc)
        os.rename(old_img_loc, new_img_loc)

    print("Training data dir: ", os.path.join(DATA_DIR, "celeba", 'train'))
    print("Testing data dir: ", os.path.join(DATA_DIR, "celeba", 'test'))

    print("Script finished!")