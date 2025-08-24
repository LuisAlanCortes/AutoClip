import matplotlib.pyplot as plt

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