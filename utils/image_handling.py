from PIL import Image

def crop_image(image, bbox):
    x, y, w, h = bbox
    cropped_img = image[y:y+h, x:x+w]
    return cropped_img

def crop_and_resize(image: Image.Image, bbox, size=(128, 128)):
    """
    Crop a PIL image using bbox and resize to model input size.

    Args:
        image (PIL.Image.Image): Input image
        bbox (list or tuple): [x, y, w, h]
        size (tuple): target output size, default (128, 128)

    Returns:
        PIL.Image.Image: Cropped and resized image
    """
    x, y, w, h = map(int, bbox)
    crop = image.crop((x, y, x + w, y + h))
    crop = crop.resize(size, resample=Image.BILINEAR)
    return crop
