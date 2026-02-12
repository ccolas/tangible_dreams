import numpy as np
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont
import random


def generate_error_screen(height, factor):
    """
    Generate an error screen image with colored squares background and error message.

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        jax array: uint8 array of shape (height, width, 3) matching update() output format
    """

    # Create colored squares background
    width = int(height*factor)
    square_size = height // 5  # Size of each colored square
    bg_img = Image.new('RGB', (width, height))
    bg_pixels = bg_img.load()

    # Fill with random colored squares
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            # Generate random cool colors (avoiding too bright/harsh colors)
            r = random.randint(20, 120)
            g = random.randint(40, 140)
            b = random.randint(60, 160)
            color = (r, g, b)

            # Fill the square
            for dy in range(min(square_size, height - y)):
                for dx in range(min(square_size, width - x)):
                    if x + dx < width and y + dy < height:
                        bg_pixels[x + dx, y + dy] = color

    # Create text overlay with semi-transparent background
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Error message text
    main_text = "error: no path from input to output detected"
    sub_text = ""#"Connect boxes so as to form a path between input nodes"
    sub_text2 = ""#"(orange, left) and output nodes (red, green, or blue, right)"

    # Try to load a font, with extensive fallback options
    font_large = None
    font_small = None

    # Try various font paths and sources (Liberation fonts first - they're nice!)
    font_attempts = [
        # Liberation fonts (clean, professional)
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 54, 36),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 54, 36),
        ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 54, 36),
        # Standard fallbacks
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 54, 36),
        ("/usr/share/fonts/TTF/DejaVuSans.ttf", 54, 36),
        ("arial.ttf", 54, 36),
        # Your available fonts
        ("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 54, 36),
    ]

    for font_path, large_size, small_size in font_attempts:
        try:
            font_large = ImageFont.truetype(font_path, large_size)
            font_small = ImageFont.truetype(font_path, small_size)
            print(f"Using font: {font_path}")  # Debug info
            break
        except:
            continue

    # If all fails, use default (unfortunately small)
    if font_large is None:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        print("Using default font (small)")

    # Calculate text positioning (center of image)
    main_bbox = draw.textbbox((0, 0), main_text, font=font_large)
    sub_bbox = draw.textbbox((0, 0), sub_text, font=font_small)
    sub2_bbox = draw.textbbox((0, 0), sub_text2, font=font_small)

    main_width = main_bbox[2] - main_bbox[0]
    main_height = main_bbox[3] - main_bbox[1]
    sub_width = sub_bbox[2] - sub_bbox[0]
    sub_height = sub_bbox[3] - sub_bbox[1]
    sub2_width = sub2_bbox[2] - sub2_bbox[0]

    # Calculate positions
    main_x = (width - main_width) // 2
    main_y = (height // 2) - 40
    sub_x = (width - sub_width) // 2
    sub_y = main_y + main_height + 20
    sub2_x = (width - sub2_width) // 2
    sub2_y = sub_y + sub_height + 5

    # Draw semi-transparent background for text area
    text_padding = 30
    text_bg_x1 = min(main_x, sub_x, sub2_x) - text_padding
    text_bg_y1 = main_y - text_padding
    text_bg_x2 = max(main_x + main_width, sub_x + sub_width, sub2_x + sub2_width) + text_padding
    text_bg_y2 = sub2_y + sub_height + text_padding

    draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2],
                   fill=(0, 0, 0, 180))  # Semi-transparent black

    # Draw text
    draw.text((main_x, main_y), main_text, font=font_large, fill=(255, 255, 255, 255))
    draw.text((sub_x, sub_y), sub_text, font=font_small, fill=(200, 200, 200, 255))
    draw.text((sub2_x, sub2_y), sub_text2, font=font_small, fill=(200, 200, 200, 255))

    # Composite the overlay onto the background
    bg_img = bg_img.convert('RGBA')
    final_img = Image.alpha_composite(bg_img, overlay)
    final_img = final_img.convert('RGB')

    # Convert to numpy array then to JAX format
    img_array = np.array(final_img)  # (height, width, 3) uint8

    # Flip vertically to match your coordinate system
    img_array = np.flipud(img_array)
    return jnp.array(img_array, dtype=jnp.uint8)