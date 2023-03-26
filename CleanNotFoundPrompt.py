import os
import argparse
from pathlib import Path

def find_and_delete_txt_without_img(folder_path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    for txt_file in txt_files:
        txt_path = os.path.join(folder_path, txt_file)
        txt_name_without_ext = os.path.splitext(txt_file)[0]

        img_found = False
        for ext in img_extensions:
            img_path = os.path.join(folder_path, txt_name_without_ext + ext)
            if os.path.exists(img_path):
                img_found = True
                break

        if not img_found:
            os.remove(txt_path)
            print(f'Deleted {txt_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and delete txt files without an image with the same name in the specified folder.')
    parser.add_argument('path', type=str, help='Path to the folder to be processed')
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print('The specified path is not a directory. Please provide a valid folder path.')
    else:
        find_and_delete_txt_without_img(args.path)
