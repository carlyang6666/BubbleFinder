import json
from PIL import Image, ImageOps
import os
# Function to invert the colors of an image
def invert_image(image_path, output_path):
    """
    Invert the colors of a black-and-white or grayscale image.
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the inverted image.
    """
    # Open the image
    img = Image.open(image_path)

    # Ensure the image is in grayscale mode
    if img.mode != 'L':
        img = img.convert('L')

    # Invert the image
    inverted_img = ImageOps.invert(img)

    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    inverted_img.save(output_path)
    print(f"Inverted image saved at {output_path}")

# split a single image into many smaller pieces
def split_image(image_path, output_dir, tile_size=(512, 512)):
    """
    image_path: absolute path of the input image
    output_dir: path of the output iamges
    tile_size: shape of the image (512, 512) in this case
    """
    # open the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # calculate the total colomn and rows number to get how many output images
    tile_width, tile_height = tile_size
    cols = img_width // tile_width
    rows = img_height // tile_height

    # create the folder/ path for the output images
    os.makedirs(output_dir, exist_ok=True)

    # view every smaller images
    for row in range(rows):
        for col in range(cols):
            # define the location of smaller images in the original
            left = col * tile_width
            upper = row * tile_height
            right = left + tile_width
            lower = upper + tile_height
            bbox = (left, upper, right, lower)

            # crop smaller images
            tile = img.crop(bbox)
            
            # save smaller images
            tile_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{row}_{col}.png"
            tile_path = os.path.join(output_dir, tile_name)
            tile.save(tile_path)

# load from json file to get parameters
def crop_images_from_json(json_path, tile_size=(512, 512)):
    """
    json_path: JSON path
    tile_size: image size (512, 512) in this case
    """
    # load json file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # get image infos from json file
    for image_info in data.get("images", []):
        image_id = image_info.get("image_id")
        image_path = image_info.get("path")
        output_dir = image_info.get("output_dir")

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        print(f"Processing {image_id}...")
        split_image(image_path, output_dir, tile_size)

def invert_images_from_json(json_path):
    """
    json_path: JSON path
    """
    # load json file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # get image infos from json file
    for image_info in data.get("images", []):
        image_id = image_info.get("image_id")
        image_path = image_info.get("path")
        output_dir = image_info.get("output_dir")

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        print(f"Processing {image_id}...")
        invert_image(image_path, output_dir)



tile_size = (512, 512)  
invert_images_from_json('invert.json')
crop_images_from_json('crop.json'  , tile_size)
