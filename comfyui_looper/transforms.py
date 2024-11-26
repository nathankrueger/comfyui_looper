from PIL import ImageFile

def zoom_in(img: ImageFile, amt: float) -> ImageFile:
    init_width, init_height = img.size

    mod_width = int(amt * float(init_width))
    mod_height = int(amt * float(init_height))

    left = init_width - mod_width
    right = init_width - left
    top = init_height - mod_height
    bottom = init_height - top

    cropped = img.crop((left, top, right, bottom))
    result = cropped.resize((init_width, init_height))
    return result