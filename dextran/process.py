import os
import numpy as np
import pandas as pd
from skimage import io, color, filters, feature, transform
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Slider
from matplotlib.lines import Line2D

# Function to process an image
def process_image(image, low_threshold, high_threshold):
    # Convert to grayscale if the image is not already in grayscale
    if len(image.shape) == 3:
        gray_img = color.rgb2gray(image)
    else:
        gray_img = image
    
    # Apply Gaussian filter to smooth the image
    blurred_img = filters.gaussian(gray_img, sigma=2)
    
    # Detect edges using Canny edge detector with manual thresholds
    edges = feature.canny(blurred_img, sigma=2, low_threshold=low_threshold, high_threshold=high_threshold)
    
    return edges

# Function to rotate image and keep the original scale
def rotate_image(image, angle, original_shape):
    # Calculate the hypotenuse to determine the size of the black square
    diagonal = int(np.ceil(np.sqrt(original_shape[0]**2 + original_shape[1]**2)))
    
    # Pad the image to ensure it fits within the original dimensions after rotation
    pad_before = (diagonal - image.shape[0]) // 2
    pad_after = diagonal - image.shape[0] - pad_before
    padded_image = np.pad(image, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')
    
    # Rotate the padded image
    rotated_image = transform.rotate(padded_image, angle, resize=True)
    
    # Crop back to the original size
    center = np.array(rotated_image.shape) // 2
    cropped_rotated_image = rotated_image[
        center[0] - original_shape[0] // 2 : center[0] + original_shape[0] // 2,
        center[1] - original_shape[1] // 2 : center[1] + original_shape[1] // 2
    ]
    
    return cropped_rotated_image

# Function to find the neck radius based on user-defined column
def find_neck_radius(image, x_column):
    vertical_line = image[:, x_column]
    white_pixels = np.where(vertical_line > 0)[0]
    if len(white_pixels) > 1:
        neck_radius = white_pixels[-1] - white_pixels[0]
        return neck_radius, white_pixels[0], white_pixels[-1]
    return 0, 0, 0

# Directory containing TIF files
folder_path = '/Users/amalli/Desktop/droplet-analysis/dextran/40x_2x2_5/'
original_folder_name = os.path.basename(os.path.normpath(folder_path))

# Get a list of all TIF files in the folder and sort them by file name
tif_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])

# Load all TIF files and add them to the image collection
image_collection = []
for file in tif_files:
    image_path = os.path.join(folder_path, file)
    image = io.imread(image_path)
    image_collection.append(image)

# Function to select ROI
def line_select_callback(eclick, erelease):
    pass  # Callback function for RectangleSelector (not used here)

def onkeypress(event):
    global roi, toggle_selector
    if event.key == 'enter':
        roi = toggle_selector.extents
        plt.close()

# Display the first image and select ROI
fig, ax = plt.subplots()
ax.imshow(image_collection[0], cmap='gray')
roi = (0, 0, 0, 0)
toggle_selector = RectangleSelector(ax, line_select_callback,
                                    useblit=True,
                                    button=[1], minspanx=5, minspany=5,
                                    spancoords='pixels', interactive=True)

# Connect the key press event to the onkeypress function
plt.connect('key_press_event', onkeypress)
plt.show()

# Ensure valid ROI dimensions
x1, x2, y1, y2 = map(int, roi)
if x1 == x2 or y1 == y2:
    raise ValueError("Invalid ROI selected. Please select a valid region of interest.")

# Crop all images based on the selected ROI
cropped_images = [img[y1:y2, x1:x2] for img in image_collection]
original_shape = (y2 - y1, x2 - x1)

# Process each cropped image to detect edges
low_threshold = 0.0001
high_threshold = 0.0004

processed_cropped_img_collection = [process_image(img, low_threshold, high_threshold) for img in cropped_images]

# Interactive rotation selection
class InteractiveRotate:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.angle = 0
        
        # Display the image
        self.im = ax.imshow(self.image, cmap='gray')
        self.slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(self.slider_ax, 'Angle', 0.0, 360.0, valinit=0.0)
        self.slider.on_changed(self.update)
        
    def update(self, val):
        self.angle = self.slider.val
        rotated_image = rotate_image(self.image, self.angle, original_shape)
        self.im.set_data(rotated_image)
        self.ax.figure.canvas.draw_idle()

# Display the first processed and cropped image and interactively rotate
fig, ax = plt.subplots()
interactive_rotate = InteractiveRotate(ax, processed_cropped_img_collection[0])
plt.show()

rotation_angle = interactive_rotate.angle

# Apply the rotation to all processed and cropped images
rotated_processed_cropped_img_collection = [rotate_image(img, rotation_angle, original_shape) for img in processed_cropped_img_collection]

# Function to select the vertical axis interactively
class InteractiveAxisSelector:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.axis_x = None
        self.line = None
        
        # Display the image
        self.im = ax.imshow(self.image, cmap='gray')
        self.cid = self.im.figure.canvas.mpl_connect('button_press_event', self.onclick)
    
    def onclick(self, event):
        if event.inaxes == self.ax:
            self.axis_x = int(event.xdata)
            if self.line:
                self.line.remove()
            self.line = Line2D([self.axis_x, self.axis_x], [0, self.image.shape[0]], color='r')
            self.ax.add_line(self.line)
            self.ax.figure.canvas.draw()
            plt.close()

# Display the first image and select vertical axis
fig, ax = plt.subplots()
axis_selector = InteractiveAxisSelector(ax, rotated_processed_cropped_img_collection[0])
plt.show()

# Calculate neck radius and store the indices of the first and last non-black pixels
neck_radii = []
first_non_black_indices = []
last_non_black_indices = []

for img in rotated_processed_cropped_img_collection:
    neck_radius, first_non_black, last_non_black = find_neck_radius(img, axis_selector.axis_x)
    neck_radii.append(neck_radius)
    first_non_black_indices.append(first_non_black)
    last_non_black_indices.append(last_non_black)

# Choose starting time
start_time = 0  # Set the starting time index here

# Filter out the neck radii to remove zeros after the starting time
filtered_neck_radii = [radius for i, radius in enumerate(neck_radii) if i >= start_time and radius > 1 or i==start_time]

# Create a new folder for all processed files and plots
output_folder = f'processed_{original_folder_name}'
os.makedirs(output_folder, exist_ok=True)

# Save neck radius vs. time data to CSV
time_indices = range(start_time, start_time + len(filtered_neck_radii))
data = {'Time (Image Index)': time_indices, 'Neck Radius (pixels)': filtered_neck_radii}
df = pd.DataFrame(data)
csv_path = os.path.join(output_folder, 'neck_radius_vs_time.csv')
df.to_csv(csv_path, index=False)

# Plot neck radius as a function of time (image index)
plt.figure()
plt.plot(time_indices, filtered_neck_radii, marker='o')
plt.title('Neck Radius as a Function of Time')
plt.xlabel('Time (Image Index)')
plt.ylabel('Neck Radius (pixels)')
plot_path = os.path.join(output_folder, 'neck_radius_vs_time_plot.png')
plt.savefig(plot_path)
plt.show()

# Plot neck radius as a function of time (image index) in loglog
plt.figure()
plt.loglog(time_indices, filtered_neck_radii, marker='o')
plt.title('Neck Radius as a Function of Time (Log-Log)')
plt.xlabel('Time (Image Index)')
plt.ylabel('Neck Radius (pixels)')
loglog_plot_path = os.path.join(output_folder, 'neck_radius_vs_time_loglog_plot.png')
plt.savefig(loglog_plot_path)
plt.show()

# Plot neck radius as a function of time (image index) in semilogx
plt.figure()
plt.semilogx(time_indices, filtered_neck_radii, marker='o')
plt.title('Neck Radius as a Function of Time (Semi-Logx)')
plt.xlabel('Time (Image Index)')
plt.ylabel('Neck Radius (pixels)')
semilogx_plot_path = os.path.join(output_folder, 'neck_radius_vs_time_semilogx_plot.png')
plt.savefig(semilogx_plot_path)
plt.show()

# Create a new folder for the images with red lines
images_output_folder = os.path.join(output_folder, 'processed_images_with_lines')
os.makedirs(images_output_folder, exist_ok=True)

# Create a new image collection with the red line drawn
for i, img in enumerate(rotated_processed_cropped_img_collection):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    if neck_radii[i] > 0:
        ax.plot([axis_selector.axis_x, axis_selector.axis_x], [first_non_black_indices[i], last_non_black_indices[i]], color='r')
    ax.set_title(f'Rotated Processed Image {i+1}')
    ax.axis('off')  # Hide axis
    
    # Save the figure to the new folder
    output_path = os.path.join(images_output_folder, f'image_with_line_{i+1}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

