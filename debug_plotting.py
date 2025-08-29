import matplotlib.pyplot as plt
import cv2
import os
import re

def plot(row, col, index, text, frame):
    plt.figure(figsize=(8, 5))
    plt.subplot(row, col, index)
    plt.title(text)
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()

def plot_marker_zone(frame):
    plt.figure(figsize=(8, 5))
    plt.title('Marker Zone')
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_tm_result(original, marker_coordinates, marker, top_left, result):
    plt.figure(figsize=(12, 8))
    # Original image with rectangles
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    img_copy = original.copy()
    x_start, y_start, w, h = marker_coordinates
    cv2.rectangle(img_copy, (x_start, y_start), (x_start + w, y_start + h), (255, 0, 0), 2)
    cv2.rectangle(img_copy, top_left, (top_left[0] + marker.shape[1], top_left[1] + marker.shape[0]), (0, 255, 0), 2)
    plt.imshow(img_copy)
    plt.axis('off')
    # Matching result
    plt.subplot(1, 2, 2)
    plt.title('Matching Result')
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def normalize_filename(filename):
    """
    Normalize filename by removing spaces and special characters.

    Args:
        filename (str): Original filename

    Returns:
        str: Normalized filename
    """
    # Remove file extension and directory path
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # Replace spaces and special chars with underscores, convert to lowercase
    normalized = re.sub(r'[^a-zA-Z0-9]', '_', base_name).lower()
    # Remove consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    return normalized


def save_tm_debug_frame(original_frame, processed_frame, marker_template, frame_number,
                       window_locations, max_loc, max_val, marker_coordinates, video_name, output_dir='debug'):
    # Normalize video name for directory
    video_dir_name = normalize_filename(video_name)
    frames_dir = os.path.join(output_dir, f'frames_{video_dir_name}')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Reduced figure size from (15, 12) to (8, 8)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Original frame with marker zone
    axes[0, 0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    x, y, w, h = marker_coordinates
    axes[0, 0].add_patch(plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2))
    axes[0, 0].set_title('Original Frame with Marker Zone', fontsize=8)
    axes[0, 0].axis('off')

    # Processed frame with sliding windows
    axes[0, 1].imshow(processed_frame, cmap='gray')
    for wx, wy in window_locations:
        axes[0, 1].add_patch(plt.Rectangle((wx, wy),
                                         marker_template.shape[1],
                                         marker_template.shape[0],
                                         fill=False, color='yellow', alpha=0.3))
    if max_loc:
        axes[0, 1].add_patch(plt.Rectangle(max_loc,
                                         marker_template.shape[1],
                                         marker_template.shape[0],
                                         fill=False, color='green', linewidth=2))
    axes[0, 1].set_title('Processed Frame with Search Windows', fontsize=8)
    axes[0, 1].axis('off')

    # Template
    axes[1, 0].imshow(marker_template, cmap='gray')
    axes[1, 0].set_title('Marker Template', fontsize=8)
    axes[1, 0].axis('off')

    # Best match
    if max_loc:
        match_region = processed_frame[max_loc[1]:max_loc[1]+marker_template.shape[0],
                                     max_loc[0]:max_loc[0]+marker_template.shape[1]]
        axes[1, 1].imshow(match_region, cmap='gray')
        axes[1, 1].set_title(f'Best Match (Score: {max_val:.3f})', fontsize=8)
    else:
        axes[1, 1].set_title('No Match Found', fontsize=8)
    axes[1, 1].axis('off')

    # Reduced main title font size from 12 to 10
    fig.suptitle(f'Frame {frame_number}\nTemplate Matching Analysis', fontsize=10)

    save_path = os.path.join(frames_dir, f'debug_frame_tm_{frame_number:06d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def save_debug_frame(original_crop, processed_image, frame_number, ocr_text, video_name, output_dir='debug'):
    # Normalize video name for directory
    video_dir_name = normalize_filename(video_name)
    frames_dir = os.path.join(output_dir, f'frames_{video_dir_name}')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Changed to horizontal layout with 1x2 grid and wider figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original frame with marker zone highlighted
    axes[0].imshow(cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Frame - Full Marker Zone')
    axes[0].add_patch(plt.Rectangle((0, 0), original_crop.shape[1], original_crop.shape[0],
                                  fill=False, color='red', linewidth=2))
    axes[0].axis('off')

    # Plot processed frame
    axes[1].imshow(processed_image, cmap='gray')
    axes[1].set_title('Processed Frame - OCR Analysis Zone')
    axes[1].axis('off')

    # Adjusted title position for horizontal layout
    fig.suptitle(f'Frame {frame_number}\nOCR Analysis of Full Marker Zone\nDetected Text: "{ocr_text}"',
                 fontsize=8, y=0.98)

    save_path = os.path.join(frames_dir, f'debug_frame_ocr_{frame_number:06d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def create_html_report(video_name, base_dir='debug'):
    """
    Generates HTML report for debug frames in video-specific directory.
    """
    video_dir_name = normalize_filename(video_name)
    image_dir = os.path.join(base_dir, f'frames_{video_dir_name}')
    output_file = os.path.join(base_dir, f'frames_report_{video_dir_name}.html')

    if not os.path.isdir(image_dir):
        print(f"Error: Directory '{image_dir}' not found.")
        return

    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    except FileNotFoundError:
        print(f"Error: Could not access directory '{image_dir}'.")
        return

    if not image_files:
        print(f"No debug images found in '{image_dir}'.")
        return

    def get_frame_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1

    image_files.sort(key=get_frame_number)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Debug Report - {os.path.basename(video_name)}</title>
        <style>
            body {{ font-family: sans-serif; background-color: #f0f0f0; }}
            .container {{ max-width: 900px; margin: auto; padding: 20px; }}
            .frame {{ border: 1px solid #ccc; margin-bottom: 20px; background-color: #fff; padding: 10px; border-radius: 5px; }}
            .frame h3 {{ margin-top: 0; }}
            .frame img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Debug Report - {os.path.basename(video_name)}</h1>
    """

    for image_file in image_files:
        # Use relative path for images
        relative_path = os.path.join('frames_' + video_dir_name, image_file).replace('\\', '/')
        title = os.path.splitext(image_file)[0].replace('_', ' ').title()
        html_content += f"""
            <div class="frame">
                <h3>{title}</h3>
                <img src="{relative_path}" alt="{title}">
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully created report: '{os.path.abspath(output_file)}'")
    except IOError as e:
        print(f"Error writing to file: {e}")