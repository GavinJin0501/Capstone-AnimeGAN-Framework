import os

import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm

from models.anime_GAN import Generator
from utils.common import load_weights, read_image
from utils.image_processing import normalize_input, resize_image, denormalize_input

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_FORMATS = {"jpeg", "jpg", "jpe", "png", "bmp"}


class Transformer:
    def __init__(self, weight="hayao", add_mean=False):
        self.G = Generator().to(DEVICE)
        load_weights(self.G, weight)
        self.G.eval()
        print("Weight loaded, ready to predict")

    def transform(self, image):
        """
        Transform a image to another style
        @Arguments:
            - image: np.array, shape = (Batch, width, height, channels)
        @Returns:
            - styled version of image: np.array
        """
        with torch.no_grad():
            fake = self.G(self.preprocess_images(image))
            fake = fake.fetach().cpu().numpy()
            # Channel last
            fake = fake.transpose(0, 2, 3, 1)
            return fake

    def transform_file(self, file_path, save_path):
        if not save_path.endswith("png"):
            raise ValueError(save_path + " should be png format")
        
        image = read_image(file_path)
        if image is None:
            raise ValueError("Could not get image from " + file_path)

        styled_img = self.transform(resize_image(image))[0]
        styled_img = denormalize_input(styled_img, dtype=np.int16)
        cv.imwrite(save_path, styled_img[..., ::-1])
        print("Styled image saved to " + save_path)

    def transform_in_dir(self, img_dir, dest_dir, max_images=0, img_size=(256, 256)):
        """
        Read all images from img_dir, transform and write the result to dest_dir
        """
        os.makedirs(dest_dir, exist_ok=True)

        files = os.listdir(img_dir)
        files = [f for f in files if self.is_valid_file(f)]
        print("Found %d images in %s" % (len(files), img_dir))

        if max_images:
            files = files[:max_images]
        
        for fname in tqdm(files):
            image = cv.imread(os.path.join(img_dir, fname))[:, :, ::-1]
            image = resize_image(image)
            styled_img = self.transform(image)[0]
            ext = fname.split(".")[-1]
            fname = fname.replace("." + ext, "")
            styled_img = denormalize_input(styled_img, dtype=np.int16)
            cv.imwrite(os.path.join(dest_dir, fname + "_anime.jpg"), styled_img[..., ::-1])
        
    def preprocess_images(self, images):
        """
        Preprocess image for inference
        @Arguments:
            - images: np.ndarray
        @Returns
            - images: torch.tensor
        """
        images = images.astype(np.float32)

        # Normalize to [-1, 1]
        images = normalize_input(images)
        images = torch.from_numpy(images).to(DEVICE)

        # Add batch dim
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # Channel first
        images = images.permute(0, 3, 1, 2)

        return images

    @staticmethod
    def is_valid_file(fname: str):
        ext = fname.split(".")[-1]
        return ext in VALID_FORMATS
