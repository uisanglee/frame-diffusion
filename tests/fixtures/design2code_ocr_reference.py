# Reference functions from NoviScl/Design2Code, metrics/ocr_free_utils.py.
# https://github.com/NoviScl/Design2Code/blob/main/Design2Code/metrics/ocr_free_utils.py
# Retrieved 2026-11-05; kept unoptimized for exact-output regression tests.
import cv2
import numpy as np
from PIL import Image

def similar(n1, n2):
    if abs(n1 - n2) <= 8:
        return True
    else:
        return False

def find_different_pixels(image1_path, image2_path):
    # Open the images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Ensure both images are of the same size
    if img1.size != img2.size:
        print(f"[Warning] Images are not the same size, {image1_path}, {image2_path}")
        return None

    # Convert images to RGB if they are not
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    # Get pixel data
    pixels1 = img1.load()
    pixels2 = img2.load()

    # List to store coordinates of different pixels
    different_pixels = []

    # Iterate through each pixel
    for x in range(img1.size[0]):
        for y in range(img1.size[1]):
            # Compare pixel colors
            r1, g1, b1 = pixels1[x, y]
            r2, g2, b2 = pixels2[x, y]
            if similar((r1 + 50) % 256, r2) and similar((g1 + 50) % 256, g2) and similar((b1 + 50) % 256, b2):
                different_pixels.append((y, x))

    if len(different_pixels) > 0:
        return np.stack(different_pixels)
    else:
        return None

def average_color(image_path, coordinates):
    """
    Calculates the average color of the specified coordinates in the given image.

    :param image: A PIL Image object.
    :param coordinates: A 2D numpy array of coordinates, where each row represents [x, y].
    :return: A tuple representing the average color (R, G, B).
    """
    # Convert image to numpy array
    image_array = np.array(Image.open(image_path).convert('RGB'))

    # Extract colors at the specified coordinates
    colors = [image_array[x, y] for x, y in coordinates]

    # Calculate the average color
    avg_color = np.mean(colors, axis=0)

    return tuple(avg_color.astype(int))

def get_blocks_from_image_diff_pixels(image_path, html_text_color_tree, different_pixels):
    image = cv2.imread(image_path)
    x_w = image.shape[0]
    y_w = image.shape[1]

    def hex_to_bgr(hex_color):
        """
        Converts a hex color string to a BGR color tuple.
        """
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb[::-1]


    def get_intersect(arr1, arr2):
        # Reshape the arrays to 1D
        arr1_reshaped = arr1.view([('', arr1.dtype)] * arr1.shape[1])
        arr2_reshaped = arr2.view([('', arr2.dtype)] * arr2.shape[1])

        # Find the intersection
        common_rows = np.intersect1d(arr1_reshaped, arr2_reshaped)

        # Reshape the result back to 2D, if needed
        common_rows = common_rows.view(arr1.dtype).reshape(-1, arr1.shape[1])
        return common_rows


    blocks = []
    for item in html_text_color_tree:
        try:
            color = np.array(hex_to_bgr(item[1]), dtype="uint8")
        except:
            continue
        
        lower = color - 4
        upper = color + 4

        mask = cv2.inRange(image, lower, upper)

        coords = np.column_stack(np.where(mask > 0))

        coords = get_intersect(coords, different_pixels)

        if coords.size == 0:
            continue

        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        color = average_color(image_path.replace("_p.png", ".png"), coords)

        blocks.append({'text': item[0].lower(), 'bbox': (y_min / y_w, x_min / x_w, (y_max - y_min + 1) / y_w, (x_max - x_min + 1) / x_w), 'color': color})
    return blocks
