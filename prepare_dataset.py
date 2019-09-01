# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:56:36 2019

@author: Rupert.Thomas

prepare_dataset.py - for training DCGAN to generate cat faces

Download Thomas Simonini's Cat dataset:
 - download .zip file
 - decompress
 - collate the files into a single folder
 - remove any outliers, unusual images, ones where the cat is at a funny angle
 - separate by size
 - crop to face

Based on Bash script taken from Alexia Jolicoeur Martineau's work https://ajolicoeur.wordpress.com/cats/
Image pre-processing is a modified version of https://github.com/microe/angora-blue/blob/master/cascade_training/describe.py by Erik Hovland

"""

# Standard library
import os
import sys
import urllib.request
import zipfile
import shutil
import glob

# Other packages
import cv2
from tqdm import tqdm

# Modules
from cat_image_preprocessing import preprocessCatFace

# Dataset files source
url = 'http://www.simoninithomas.com/data/cats.zip'
zip_filename = 'cats.zip'  # N.B. there are actually two files, one inside the other

temp_folder = 'dataset_temp'

dataset_root_folder = 'cat_dataset'
images1_path = os.path.join(dataset_root_folder,'cats_bigger_than_64x64')
images2_path = os.path.join(dataset_root_folder,'cats_bigger_than_128x128')
    
# Make folder structure
for folder_path in [temp_folder, dataset_root_folder, images1_path, images2_path]:
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

#%% Download Cats dataset, if required
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if not os.path.exists(zip_filename):
    print('Beginning file download:')
    
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=zip_filename, reporthook=t.update_to)

## Unzip dataset
print('Unzipping files...')
# First ZIP file
try:
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(temp_folder)
except:
    sys.exit('Failed the first half of unzipping the dataset')

# Second Zip file
try:
    second_zip_file_location = os.path.join(temp_folder, zip_filename)
    with zipfile.ZipFile(second_zip_file_location, 'r') as zip_ref:
        zip_ref.extractall(temp_folder)
except:
    sys.exit('Failed the second half of unzipping the dataset')

## Copy files
print('Analysing files...')
# Removing outliers
# Corrupted, drawings, badly cropped, inverted, impossible to tell it's a cat, blocked face
image_ignore_list = ['rm 00000004_007.jpg', '00000007_002.jpg', '00000045_028.jpg', 
                     '00000050_014.jpg', '00000056_013.jpg', '00000059_002.jpg', 
                     '00000108_005.jpg', '00000122_023.jpg', '00000126_005.jpg', 
                     '00000132_018.jpg', '00000142_024.jpg', '00000142_029.jpg', 
                     '00000143_003.jpg', '00000145_021.jpg', '00000166_021.jpg', 
                     '00000169_021.jpg', '00000186_002.jpg', '00000202_022.jpg', 
                     '00000208_023.jpg', '00000210_003.jpg', '00000229_005.jpg', 
                     '00000236_025.jpg', '00000249_016.jpg', '00000254_013.jpg', 
                     '00000260_019.jpg', '00000261_029.jpg', '00000265_029.jpg', 
                     '00000271_020.jpg', '00000282_026.jpg', '00000316_004.jpg', 
                     '00000352_014.jpg', '00000400_026.jpg', '00000406_006.jpg', 
                     '00000431_024.jpg', '00000443_027.jpg', '00000502_015.jpg', 
                     '00000504_012.jpg', '00000510_019.jpg', '00000514_016.jpg', 
                     '00000514_008.jpg', '00000515_021.jpg', '00000519_015.jpg', 
                     '00000522_016.jpg', '00000523_021.jpg', '00000529_005.jpg', 
                     '00000556_022.jpg', '00000574_011.jpg', '00000581_018.jpg', 
                     '00000582_011.jpg', '00000588_016.jpg', '00000588_019.jpg', 
                     '00000590_006.jpg', '00000592_018.jpg', '00000593_027.jpg', 
                     '00000617_013.jpg', '00000618_016.jpg', '00000619_025.jpg', 
                     '00000622_019.jpg', '00000622_021.jpg', '00000630_007.jpg', 
                     '00000645_016.jpg', '00000656_017.jpg', '00000659_000.jpg', 
                     '00000660_022.jpg', '00000660_029.jpg', '00000661_016.jpg', 
                     '00000663_005.jpg', '00000672_027.jpg', '00000673_027.jpg', 
                     '00000675_023.jpg', '00000692_006.jpg', '00000800_017.jpg', 
                     '00000805_004.jpg', '00000807_020.jpg', '00000823_010.jpg', 
                     '00000824_010.jpg', '00000836_008.jpg', '00000843_021.jpg', 
                     '00000850_025.jpg', '00000862_017.jpg', '00000864_007.jpg', 
                     '00000865_015.jpg', '00000870_007.jpg', '00000877_014.jpg', 
                     '00000882_013.jpg', '00000887_028.jpg', '00000893_022.jpg', 
                     '00000907_013.jpg', '00000921_029.jpg', '00000929_022.jpg', 
                     '00000934_006.jpg', '00000960_021.jpg', '00000976_004.jpg', 
                     '00000987_000.jpg', '00000993_009.jpg', '00001006_014.jpg', 
                     '00001008_013.jpg', '00001012_019.jpg', '00001014_005.jpg', 
                     '00001020_017.jpg', '00001039_008.jpg', '00001039_023.jpg', 
                     '00001048_029.jpg', '00001057_003.jpg', '00001068_005.jpg', 
                     '00001113_015.jpg', '00001140_007.jpg', '00001157_029.jpg', 
                     '00001158_000.jpg', '00001167_007.jpg', '00001184_007.jpg', 
                     '00001188_019.jpg', '00001204_027.jpg', '00001205_022.jpg', 
                     '00001219_005.jpg', '00001243_010.jpg', '00001261_005.jpg', 
                     '00001270_028.jpg', '00001274_006.jpg', '00001293_015.jpg', 
                     '00001312_021.jpg', '00001365_026.jpg', '00001372_006.jpg', 
                     '00001379_018.jpg', '00001388_024.jpg', '00001389_026.jpg', 
                     '00001418_028.jpg', '00001425_012.jpg', '00001431_001.jpg', 
                     '00001456_018.jpg', '00001458_003.jpg', '00001468_019.jpg', 
                     '00001475_009.jpg', '00001487_020.jpg', '00000003_019.jpg']

# generate list of source/destination copy pairs
file_copy_src_dest = []
dataset_subfolders = [name for name in os.listdir(temp_folder) if os.path.isdir(os.path.join(temp_folder, name))]
for subfolder in dataset_subfolders:
    this_folder_path = os.path.join(temp_folder, subfolder)
    src_files = os.listdir(this_folder_path)
    for file_name in src_files:
        # get image files
        src_filepath = os.path.join(this_folder_path, file_name)
        if os.path.isfile(src_filepath) and file_name.endswith('.jpg') and (file_name not in image_ignore_list):
            # check if datafile also exists
            data_filename = file_name + '.cat'
            src_data_filepath = os.path.join(this_folder_path, data_filename)
            if os.path.isfile(src_data_filepath):
                # add data file to copy list
                dest_data_filepath = os.path.join(dataset_root_folder, data_filename)
                file_copy_src_dest.append((src_data_filepath, dest_data_filepath))
                
                # add image file to copy list
                dest_img_filepath = os.path.join(dataset_root_folder, file_name)
                file_copy_src_dest.append((src_filepath, dest_img_filepath))

# copy files with progress bar
print('Copying files...')
for src, dest in tqdm(file_copy_src_dest):
    shutil.copyfile(src, dest)
    
#%% Process image files
print('Running image pre-processing...')
#cat_image_preprocessing.describePositive()
image_files = glob.glob(os.path.join(dataset_root_folder, '*.jpg'))
with tqdm(total=len(image_files)) as pbar:
	for c,imagePath in enumerate(image_files):

		# Open the '.cat' annotation file associated with this image.
		input = open('%s.cat' % imagePath, 'r')
		# Read the coordinates of the cat features from the file. 
        # Discard the first number, which is the number of features.
		coords = [int(i) for i in input.readline().split()[1:]]
		# Read the image.
		image = cv2.imread(imagePath)

		# Straighten and crop the cat face.
		crop = preprocessCatFace(coords, image)

		if crop is None:
			print('Failed to preprocess image at {}'.format(imagePath), file=sys.stderr)
			continue

		# Save the crop to folders based on size
		h, w, colors = crop.shape
		if min(h,w) >= 64:
			Path1 = imagePath.replace(dataset_root_folder, images1_path)
			cv2.imwrite(Path1, crop)
		if min(h,w) >= 128:
			Path2 = imagePath.replace(dataset_root_folder, images2_path)
			cv2.imwrite(Path2, crop)
		pbar.update(1)

#%%
        
# remove temporary folder and contents
print('Cleaning up...')
try:
    shutil.rmtree(temp_folder)
except:
    print('Unable to remove temporary directory: {}'.format(temp_folder))
    
# remove ZIP file
try:
    os.remove(zip_filename)
except:
    print('Unable to remove downloaded ZIP file: {}'.format(zip_filename))


print('Done!')
