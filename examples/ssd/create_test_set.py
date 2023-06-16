# Create test set with correct annotations and images for VOC12 test set

import argparse
import os
import glob
import logging
import shutil

from pathlib import Path


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        logging.info(f"Created folder at `{path}`")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--dst_path', type=str)

    parsed_args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    src_path = parsed_args.src_path
    dst_path = parsed_args.dst_path

    annotations_folder = os.path.join(src_path, "Annotations")
    annotations = glob.glob(os.path.join(annotations_folder, "*.xml"))
    annotations = [Path(a).stem for a in annotations]

    image_folder = os.path.join(src_path, "JPEGImages")

    # Create output folders
    op_parent_folder = os.path.join(dst_path, 'VOC2012_test')
    op_annotations_folder = os.path.join(op_parent_folder, 'Annotations')
    op_images_folder = os.path.join(op_parent_folder, 'JPEGImages')
    op_sets_folder = os.path.join(op_parent_folder, 'ImageSets')
    op_sets_main_folder = os.path.join(op_sets_folder, 'Main')
    for p in [op_parent_folder, op_annotations_folder, op_images_folder, 
              op_sets_folder, op_sets_main_folder]:
        create_folder(p)

    correct_annotations = []

    for i, annotation in enumerate(annotations):
        src_xml_path = os.path.join(annotations_folder, annotation + ".xml")
        src_img_path = os.path.join(image_folder, annotation + ".jpg")
        
        if not os.path.isfile(src_img_path):
            logging.warning(f"No image for xml annotation: {annotation}")
            continue
        
        shutil.copy(src_xml_path, op_annotations_folder)
        shutil.copy(src_img_path, op_images_folder)
        logging.info(f"[{i}] Copied sample: {annotation}")

        correct_annotations.append(annotation)

    op_samples_file = os.path.join(op_sets_main_folder, "test.txt")
    with open(op_samples_file, mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(correct_annotations))

    logging.info(f"Written to text file: `{op_samples_file}`")


if __name__ == "__main__":
    main()