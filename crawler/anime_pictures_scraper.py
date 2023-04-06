import os
import time
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup


def get_post_links(tag, page):
    """
    Get links to posts for a given tag and page number.
    """
    base_url = "https://anime-pictures.net"
    url = f"{base_url}/posts?page={page}&search_tag={tag}&order_by=date&ldate=0&lang=en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    post_blocks = soup.find_all(class_="img_block_big")
    return [base_url + post_block.find("a")["href"] for post_block in post_blocks]


def get_full_image_link(post_link):
    """
    Get the direct link to the full-size image for a given post.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(post_link, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    download_icon = soup.find("a", class_="download_icon")
    return download_icon["href"]


def download_images(tag, max_pages, start_page=0, end_page=None, delay=3):
    """
    Download all images for a given tag and page range.
    """
    output_dir = os.path.join("data", tag)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if end_page is None:
        end_page = max_pages
    else:
        end_page = min(end_page, max_pages)

    for page in range(start_page, end_page):
        try:
            post_links = get_post_links(tag, page)

            for post_link in tqdm(post_links, desc=f"Downloading images (Page {page + 1}/{max_pages})"):
                try:
                    full_image_link = get_full_image_link(post_link)
                    file_name = os.path.basename(full_image_link)
                    file_name_no_ext, file_ext = os.path.splitext(file_name)
                    file_name_no_ext = file_name_no_ext[:150] if len(file_name_no_ext) > 150 else file_name_no_ext
                    file_name = file_name_no_ext + file_ext     
                    output_path = os.path.join(output_dir, file_name)

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                    response = requests.get(full_image_link, headers=headers, stream=True, timeout=10)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download image: {e}")
                time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"Failed to get post links: {e}")


if __name__ == "__main__":
    tag = "girl"
    max_pages = 5000
    download_images(tag, max_pages, start_page=0, end_page=200)

