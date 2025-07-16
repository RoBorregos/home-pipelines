workdir = "/home/roborregos/robocup-2025/"

from PIL import Image
import os


def verify_or_create_dir(path):
    os.makedirs(path, exist_ok=True)


results_path = workdir + "png_dataset/"

# Get all subfolders inside results_path
path_to_classes = [f.path for f in os.scandir(
    results_path) if f.is_dir() and "bbs" not in f.name]

# Build list of (path, class_name) tuples
fg_folders = [(path, os.path.basename(path)) for path in path_to_classes]

# Define folders
bg_folder = workdir + "bg/"
verify_or_create_dir(bg_folder)
output_folder = workdir + "AAAA2/"
objects_list = [os.path.basename(class_path) for class_path in path_to_classes]

# If you have a list of original classes, uncomment and fill it
original_classes = [
    #     "exampleClass1", "exampleClass2", "exampleClass3", "exampleClass4",
]

# Add new classes at the end only
all_detected_classes = [os.path.basename(
    class_path) for class_path in path_to_classes]

# Append only the new classes
final_classes = original_classes.copy()
for cls in all_detected_classes:
    if cls not in final_classes:
        final_classes.append(cls)

# Create annotations_ID and categories using final_classes
annotations_ID = {cls: i for i, cls in enumerate(final_classes)}
categories = [{"id": i, "name": cls} for i, cls in enumerate(final_classes)]

print("annotations_ID:", annotations_ID)
print("categories:", categories)

# Load the list of files in each of the folders
fg_files = {}
for folder, category in fg_folders:
    fg_files[category] = os.listdir(folder)

# Define the folder structure
subfolders = [
    "train/images",
    "train/labels",
    "test/images",
    "test/labels",
    "valid/images",
    "valid/labels",
]

# Create them
for sub in subfolders:
    verify_or_create_dir(os.path.join(output_folder, sub))
    
    

import random
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ExifTags, ImageFilter
import numpy as np
from torchvision import transforms
import tqdm
import cv2
import math
import argparse
import sys
from numba import njit, prange
import numba


# Constants for augmentations
ZOOM_FACTOR = 1.2  # Maximum zoom factor
BRIGHTNESS_FACTOR = 0.2  # Brightness adjustment range (0.5 to 1.5)
CONTRAST_FACTOR = 0.2  # Contrast adjustment range (0.5 to 1.5)
SATURATION_FACTOR = 0.2  # Saturation adjustment range (0.5 to 1.5)
HUE_FACTOR = 0.02  # Hue adjustment range (-0.5 to 0.5)
BLUR_FACTOR = 0.25  # Max Blur adjustment range (0 to blur_factor)
NOISE_PROBABILITY = 0.5  # Probability of adding noise
NOISE_FACTOR = 0.15  # Noise adjustment range (0 to noise_factor)
BLOB_COUNT_MIN = 0  # Minimum number of blobs to add
BLOB_COUNT_MAX = 2  # Maximum number of blobs to add
BLOB_SIZE = (15, 150)  # Size range of blobs (min, max)
BLOB_PROBABILITY = 0.15  # Probability of adding a blob
QUALITY_FACTOR = 0.25 # Min quality that can be applied to the image compression e.g. 0.25 means image can be compressed to 25% of its original quality 

MULTIPLIER = 5

def augment_image(image):
    
    def adjust_brightness(image):
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(1 - BRIGHTNESS_FACTOR, 1 + BRIGHTNESS_FACTOR)
        return enhancer.enhance(factor)

    def adjust_contrast(image):
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(1 - CONTRAST_FACTOR, 1 + CONTRAST_FACTOR)
        return enhancer.enhance(factor)

    def adjust_saturation(image):
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(1 - SATURATION_FACTOR, 1 + SATURATION_FACTOR)
        return enhancer.enhance(factor)

    def adjust_hue(image):
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(1 - HUE_FACTOR, 1 + HUE_FACTOR)
        # Convert to HSV, adjust hue, and convert back to RGB
        hsv_image = image.convert('HSV')
        h, s, v = hsv_image.split()
        h = h.point(lambda p: (p + int(factor * 255)) % 256)
        hsv_image = Image.merge('HSV', (h, s, v))
        return hsv_image.convert('RGB')

    def add_blur(image):
        blur_factor = random.uniform(0, BLUR_FACTOR)
        if blur_factor > 0:
            return image.filter(ImageFilter.GaussianBlur(radius=blur_factor))
        return image

    def add_noise(image):
        noise_factor = random.uniform(0, NOISE_FACTOR)
        if not random.random() < NOISE_PROBABILITY:
            return image
        if noise_factor > 0:
            img_arr = np.array(image, dtype=np.float32)
            if img_arr.shape[2] == 4:
                # Only add noise to RGB channels, keep alpha unchanged
                noise = np.random.normal(0, noise_factor, [img_arr.shape[0], img_arr.shape[1], 3])
                noise = (noise * 255).astype(np.float32)
                img_arr[:, :, :3] = np.clip(img_arr[:, :, :3] + noise, 0, 255)
                noisy_image = img_arr.astype(np.uint8)
            else:
                noise = np.random.normal(0, noise_factor, [img_arr.shape[0], img_arr.shape[1], 3])
                noise = (noise * 255).astype(np.float32)
                img_arr = np.clip(img_arr + noise, 0, 255)
                noisy_image = img_arr.astype(np.uint8)
                return Image.fromarray(noisy_image, mode=image.mode)
        return image

    def generate_blob_points(center_x, center_y, radius, irregularity=0.5, spikiness=0.5, num_points=12):
        """
        Generate a blob-like shape using a star/polygon algorithm with randomness.
        """
        
        points = []
        angle_step = 2 * math.pi / num_points

        for i in range(num_points):
            angle = i * angle_step
            # Vary radius for spikiness
            rand_radius = radius * (1 + random.uniform(-spikiness, spikiness))
            # Add offset for irregularity
            offset_angle = angle + random.uniform(-irregularity, irregularity) * angle_step
            x = center_x + rand_radius * math.cos(offset_angle)
            y = center_y + rand_radius * math.sin(offset_angle)
            points.append((x, y))

        return points

    def add_blobs(image):
        if random.random() > BLOB_PROBABILITY:
            return image
        draw = ImageDraw.Draw(image)
        blob_count = random.randint(BLOB_COUNT_MIN, BLOB_COUNT_MAX)

        for _ in range(blob_count):
            x = random.randint(0, image.width)
            y = random.randint(0, image.height)
            size = random.randint(*BLOB_SIZE)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(100, 255)  # Optional alpha
            )
            blob_points = generate_blob_points(x, y, size, irregularity=0.4, spikiness=0.6, num_points=random.randint(8, 16))
            draw.polygon(blob_points, fill=color)

        return image
    
    def alter_quality(image):
        quality_ratio = random.uniform(QUALITY_FACTOR, 1)
        frame = np.array(image)
        result, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality_ratio * 100)])
        return Image.fromarray(cv2.imdecode(encoded, cv2.IMREAD_COLOR))

    # Apply augmentations
    augmentations = [adjust_brightness, adjust_contrast, adjust_saturation, add_blobs, 
                    adjust_hue, add_blur, add_noise, alter_quality]
    random.shuffle(augmentations)
    
    for aug in augmentations:
        image = aug(image)
    
    return image

import random
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, UnidentifiedImageError
import numpy as np
import yaml
import tqdm
import multiprocessing as mp
from functools import partial

images = []
annotations = []
annotations2 = []
annot_csv = []

img_id = int(0)
anno_id = int(0)

min_size_ratio = 0.05  # Objects must be at least 10% of bg size
max_size_ratio = 0.35  # Objects can be at most 75% of bg size

# Define the maximum overlap as a percentage
max_overlap_pct = 25

trainfolder = output_folder + "train/"
validfolder = output_folder + "valid/"

images_to_generate = 50
max_objects_per_image = 8

progress_bar = tqdm.tqdm(total=images_to_generate, desc="Generating images")

def generate_single_image(img_id, fg_folders, fg_files, bg_folder, annotations_ID, 
                         objects_list, trainfolder, max_objects_per_image):
    """Generate a single synthetic image with annotations"""
    
    # Create empty label file
    with open(f'{trainfolder}labels/{img_id}.txt', 'w') as file:
        pass

    # Decide if this image should be empty
    if random.random() < 0.10:  # 10% chance
        num_objects = 0
    else:
        num_objects = random.randint(1, max_objects_per_image)

    fg_categories = random.choices(objects_list, k=num_objects)
    fg_files_selected = [[category, random.choice(fg_files[category])] for category in fg_categories]

    fg_imgs = []
    for img in fg_files_selected:
        folder = [f[0] for f in fg_folders if f[1] == img[0]][0]
        fg_img = Image.open(folder + "/" + img[1]).convert("RGBA")
        fg_imgs.append([img[0], Image.open(folder + "/" + img[1]), folder + img[1]])

    bg_files = os.listdir(bg_folder)
    bg_file = random.choice(bg_files)
    while not os.path.isfile(bg_folder + bg_file):
        bg_file = random.choice(bg_files)
    bg_img = Image.open(bg_folder + bg_file)
    bg_img = bg_img.convert("RGBA")

    occupied_mask = np.zeros((bg_img.height, bg_img.width), dtype=np.uint8)
    
    image_annotations = []
    anno_id_start = img_id * 1000  # Avoid ID conflicts
    
    for idx, img in enumerate(fg_imgs):
        fg_img = img[1]

        angle = random.randint(-5, 5)
        fg_img = fg_img.rotate(angle, resample=Image.BICUBIC, expand=True)

        # Resize using dynamic scale bounds based on background
        original_w, original_h = fg_img.size
        max_scale_w = (bg_img.width * max_size_ratio) / original_w
        max_scale_h = (bg_img.height * max_size_ratio) / original_h
        min_scale_w = (bg_img.width * min_size_ratio) / original_w
        min_scale_h = (bg_img.height * min_size_ratio) / original_h

        scale_min = max(min_scale_w, min_scale_h)
        scale_max = min(max_scale_w, max_scale_h)

        scale_min = max(scale_min, 0.01)
        scale_max = min(scale_max, 1.0)

        if scale_max < scale_min:
            scale_max = scale_min

        scale = random.uniform(scale_min, scale_max)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        fg_img = fg_img.resize((new_w, new_h), resample=Image.BICUBIC)

        fg_img = ImageEnhance.Brightness(fg_img).enhance(random.uniform(0.8, 1.2))
        fg_img = ImageEnhance.Contrast(fg_img).enhance(random.uniform(0.8, 1.2))
        fg_img = ImageEnhance.Color(fg_img).enhance(random.uniform(0.8, 1.2))
        fg_img = fg_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))

        img[1] = fg_img

        max_x = bg_img.width - fg_img.width
        max_y = bg_img.height - fg_img.height

        for attempt in range(50):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            fg_mask = np.array(fg_img.split()[-1]) > 0

            mask_h, mask_w = fg_mask.shape
            if y + mask_h > occupied_mask.shape[0] or x + mask_w > occupied_mask.shape[1]:
                continue

            occ_crop = occupied_mask[y:y + mask_h, x:x + mask_w]
            fg_mask_area = np.sum(fg_mask)
            occ_crop_area = np.sum(occ_crop)
            overlap = np.sum(np.logical_and(fg_mask, occ_crop))

            allowed_overlap_fg = max_overlap_pct / 100 * fg_mask_area
            allowed_overlap_occ = max_overlap_pct / 100 * occ_crop_area if occ_crop_area > 0 else 0

            if overlap <= allowed_overlap_fg and overlap <= allowed_overlap_occ:
                occupied_mask[y:y + fg_img.height, x:x + fg_img.width] = np.logical_or(occ_crop, fg_mask)
                break
        else:
            continue

        seg_img = fg_img
        img_arr = np.array(seg_img)
        mask = img_arr[:, :, 3] != 0

        segmentation = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    segmentation.append(j + x)
                    segmentation.append(i + y)
        segmentation = [segmentation]

        area = abs(sum(
            segmentation[0][2 * i] * segmentation[0][(2 * i + 3) % len(segmentation[0])] -
            segmentation[0][(2 * i + 2) % len(segmentation[0])] * segmentation[0][2 * i + 1]
            for i in range(len(segmentation[0]) // 2)
        )) / 2

        bg_img.paste(fg_img, (x, y), fg_img)

        x1, y1 = x, y
        x2 = x + fg_img.width
        y2 = y + fg_img.height

        if x2 <= x1 or y2 <= y1:
            continue

        x_center_ann = ((x1 + x2) / 2) / bg_img.width
        y_center_ann = ((y1 + y2) / 2) / bg_img.height
        width_ann = (x2 - x1) / bg_img.width
        height_ann = (y2 - y1) / bg_img.height

        if not (0 <= x_center_ann <= 1 and 0 <= y_center_ann <= 1 and 0 <= width_ann <= 1 and 0 <= height_ann <= 1):
            continue

        # Create segmentation labels
        normalize_seg = []
        for i in range(len(segmentation[0])):
            if i % 2 == 0:
                val = segmentation[0][i] / bg_img.width
            else:
                val = segmentation[0][i] / bg_img.height
            normalize_seg.append(str(val))
        
        with open(f'{trainfolder}labels/{img_id}.txt', 'a') as f:
            f.write(f"{annotations_ID[img[0]]} {' '.join(normalize_seg)}\n")

        # Store annotation data
        annotation_data = {
            "id": anno_id_start + idx,
            "image_id": img_id,
            "category_id": annotations_ID[img[0]],
            "bbox": [x, y, fg_img.width, fg_img.height],
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0
        }
        image_annotations.append(annotation_data)
    if len(image_annotations) == 0:
        with open(f'{trainfolder}labels/{img_id}.txt', 'w') as f:
            f.write("")
    # Apply additional augmentation
    bg_img = augment_image(bg_img)
    bg_img = bg_img.convert("RGB")
    bg_img.save(f"{trainfolder}images/{img_id}.jpg", quality=100)
    
    image_data = {
        "id": img_id,
        "file_name": f"{img_id}.jpg",
        "height": bg_img.height,
        "width": bg_img.width
    }
    
    return image_data, image_annotations

# Parallel execution
if __name__ == "__main__":
    num_processes = min(mp.cpu_count() - 1, 24)  # Leave one core free, max 8 processes
    
    # Create partial function with fixed parameters
    generate_func = partial(
        generate_single_image,
        fg_folders=fg_folders,
        fg_files=fg_files,
        bg_folder=bg_folder,
        annotations_ID=annotations_ID,
        objects_list=objects_list,
        trainfolder=trainfolder,
        max_objects_per_image=max_objects_per_image
    )
    
    progress_bar = tqdm.tqdm(total=images_to_generate, desc="Generating images")
    
    with mp.Pool(processes=num_processes) as pool:
        results = []
        for i in range(images_to_generate):
            result = pool.apply_async(generate_func, (i,))
            results.append(result)
        
        # Collect results
        for result in results:
            image_data, image_annotations = result.get()
            images.append(image_data)
            annotations.extend(image_annotations)
            annotations2.extend(image_annotations)
            progress_bar.update(1)
    
    progress_bar.close()

# Create data.yaml
data = dict(
    train=f"{trainfolder}images",
    val=f"{validfolder}images",
    test=f"{validfolder}images",
    nc=len(annotations_ID),
    names=list(annotations_ID.keys())
)

with open(f'{output_folder}data.yaml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)