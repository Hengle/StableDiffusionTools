import os
import sys
import requests
from PIL import Image
from bs4 import BeautifulSoup

def get_image_ids(tag, page):
    url = f'https://danbooru.donmai.us/posts.json?tags={tag}&page={page}&limit=200'
    response = requests.get(url)
    data = response.json()

    return [str(post['id']) for post in data]

def get_image_tags(image_id):
    url = f'https://danbooru.donmai.us/posts/{image_id}.json'
    response = requests.get(url)
    data = response.json()

    return data['tag_string'].split(' ')
    
def download_image(image_id, save_path):
    url = f'https://danbooru.donmai.us/posts/{image_id}.json'
    response = requests.get(url)
    data = response.json()

    # Check if file_url exists
    if 'file_url' not in data:
        print(f'Image {image_id} has no file_url, skipping.')
        return None

    # Get image tags
    tags = data['tag_string'].split(' ')

    # Get file extension
    file_ext = data['file_ext']

    # Update save_path with the correct file extension
    save_path = os.path.splitext(save_path)[0] + '.' + file_ext

    # Get the image URL
    file_url = data['file_url']

    # Download the image
    response = requests.get(file_url)

    with open(save_path, 'wb') as f:
        f.write(response.content)

    # Convert to JPEG if necessary
    if file_ext not in ['jpg', 'jpeg']:
        try:
            img = Image.open(save_path)
            img = img.convert('RGB')
            save_path = os.path.splitext(save_path)[0] + '.jpg'
            img.save(save_path)
        except Exception as e:
            print(f'Error converting {save_path} to JPEG: {e}')

    # Save tags in a text file
    tag_file = os.path.splitext(save_path)[0] + '.txt'
    with open(tag_file, 'w') as f:
        f.write(', '.join(tags))

    # Return tags as a comma-separated string
    tag_str = ', '.join(tags)
    return tag_str

def download_artist_images(artist_tag, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    page = 1
    while True:
        image_ids = get_image_ids(artist_tag, page)
        if not image_ids:
            break

        for image_id in image_ids:
            save_path = os.path.join(output_dir, f'{image_id}.jpg')
            tags = download_image(image_id, save_path)
            print(f'Downloaded {image_id}.jpg with {tags}')

        page += 1


if __name__ == '__main__':
    artist_tag = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = artist_tag

    download_artist_images(artist_tag, output_dir)
artist_tag = 'quan_%28kurisu_tina%29'
output_dir = 'download'

download_artist_images(artist_tag, output_dir)
