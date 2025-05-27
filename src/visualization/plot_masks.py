from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils.file_operations import load_segmentation_data, read_points_from_txt

def plot_masks(segmentation_file, points_file, output_dir, pixel_to_nm=100, white_light_file=None):
    """
    Visualize the masks from the segmentation file, overlay points, and save the image.

    Parameters:
    segmentation_file (str): Path to the segmentation .npy file.
    points_file (str): Path to the .txt file containing points.
    output_dir (str): Directory where the output images will be saved.
    pixel_to_nm (int): Conversion factor from pixels to nanometers (default: 100).
    white_light_file (str, optional): Path to the white light image file to use as the background.
    """
    # Load segmentation data
    segmentation_data = load_segmentation_data(segmentation_file)
    masks = segmentation_data.get('masks', None)

    if masks is None:
        raise ValueError("No masks found in the segmentation file.")

    # Load points from the .txt file
    points = read_points_from_txt(points_file)

    # Scale points from nanometers to pixels
    points_scaled = np.column_stack((points[:, :2] / pixel_to_nm, points[:, 2]))

    # Create an output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get the unique mask IDs (excluding background, assumed to be 0)
    unique_masks = [mask_id for mask_id in np.unique(masks) if mask_id != 0]

    # Create a figure for visualization
    plt.figure(figsize=(10, 10))

    # Display the white light image as the background if provided
    if white_light_file:
        white_light_image = plt.imread(white_light_file)
        plt.imshow(white_light_image, cmap='gray', alpha=1.0)  # Display white light image
    else:
        plt.imshow(np.zeros(masks.shape), cmap='gray')  # Default gray background

    # Assign a unique color to each mask
    # color_map = plt.get_cmap("hsv", len(unique_masks) + 1)

    cmap = plt.cm.tab20
    #cmap=plt.cm.nipy_spectral

    plt.imshow(np.where(masks == 0, np.nan, masks), cmap=cmap, alpha=0.2, vmin=1, vmax=len(unique_masks))

# # Plot each mask with its corresponding color from the `colors` array
#     for mask_id, color in zip(unique_masks, colors):
#         mask = masks == mask_id
#         plt.imshow(np.where(mask, mask_id, np.nan), cmap=plt.cm.gray, alpha=0.5)
#         plt.imshow(np.where(mask, 1, np.nan), color=color, alpha=0.5)  # Use the color for the mask

    #plt.imshow(np.where(masks == 0, np.nan, masks), cmap=color_map, alpha=0.5, vmin=1, vmax=len(unique_masks))

    plt.savefig(Path(output_dir) / "masks_overlay.png", dpi=800, bbox_inches='tight')
    plt.clf()


    # Plot points with different colors for each channel
    unique_channels = np.unique(points_scaled[:, 2])
    channel_colors = plt.cm.tab10(np.linspace(0, .8, len(unique_channels)))  # Use a colormap for channels


    
    for mask_id in unique_masks:
        plt.figure(figsize=(20, 20))
        fig, ax = plt.subplots()
        for channel, color in zip(unique_channels, channel_colors):
            mask = masks == mask_id

                # Filter points for the current channel
            channel_points = points_scaled[points_scaled[:, 2] == channel]

            # Filter points to include only those within the masks
            points_in_masks = []
            for point in channel_points:
                pixel_x, pixel_y = point[:2].astype(int)
                if 0 <= pixel_x < mask.shape[1] and 0 <= pixel_y < mask.shape[0]:
                    if mask[pixel_y, pixel_x]:  # Note: y corresponds to rows, x to columns
                        points_in_masks.append(point)

            points_in_masks = np.array(points_in_masks)

            # Plot the filtered points
            if len(points_in_masks) > 0:
                #plt.scatter(points_in_masks[:, 0], points_in_masks[:, 1], s=1, c=[color], label=f"Channel {int(channel)}", alpha=0.1
                            
                ax.plot(
                    points_in_masks[:, 0],
                    points_in_masks[:, 1],
                    "o",
                    markerfacecolor=color,
                    markeredgecolor="none",
                    markersize= .1,
                    alpha = .5
                )
            #plt.scatter(channel_points[:, 0], channel_points[:, 1], s=1, c=[color], label=f"Channel {int(channel)}", alpha=0.01)

        # Display the white light image as the background if provided
        if white_light_file:
            white_light_image = plt.imread(white_light_file)
            plt.imshow(white_light_image, cmap='gray', alpha=1.0)  # Display white light image
        else:
            plt.imshow(np.zeros(masks.shape), cmap='gray')  # Default gray background

        plt.legend()
        plt.title(f"Point Visualization for mask {mask_id}")
        plt.axis('off')

        # Save the visualization image

        visualization_image_path = Path(output_dir) / f"mask_{mask_id}_point_visualization.png"
        plt.savefig(visualization_image_path, dpi=800)
        plt.close()

    print(f"Mask visualization with points saved to {visualization_image_path}")