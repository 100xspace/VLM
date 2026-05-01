def download_image(url, image_dir):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        uuid = hashlib.sha256(response.content).hexdigest()[:12]
        filename = uuid + ".jpg"
        img.save(os.path.join(image_dir, filename))
        return img.size, uuid  # returns (width, height)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None, None
