from pathlib import Path
import numpy as np
from tkinter import Tk, filedialog
from utils.file_operations import load_segmentation_data, read_points_from_txt
from visualization.plot_masks import plot_masks

testing = True  # Set to True for testing mode that doesn't ask user to select files

def select_file(file_type, file_extensions):
    """
    Open a file dialog to select a file.

    Parameters:
    file_type (str): Description of the file type (e.g., "segmentation file").
    file_extensions (list): List of allowed file extensions (e.g., [".npy"]).

    Returns:
    str: Path to the selected file.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title=f"Select the {file_type}",
        filetypes=[(file_type, file_extensions)]
    )
    if not file_path:
        raise FileNotFoundError(f"No {file_type} selected.")
    return file_path

def select_directory():
    """
    Open a file dialog to select a directory.

    Returns:
    str: Path to the selected directory.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    directory_path = filedialog.askdirectory(
        title="Select the output directory",
        initialdir=Path.cwd() # Default to the current working directory)
      ) 
    if not directory_path:
        raise FileNotFoundError("No directory selected.")

    return directory_path

def main():
    """
    Main function to process segmentation masks and save points associated with each mask.
    """
    # Step 1: Collect all files
    if(testing):
        print("Testing mode: Using default files.")
        segmentation_file = "/Users/bvossen/Documents/STORM_PALM/analyzed(John)/log_phase/white_light/20190617_dsm_white_002_seg.npy"
        points_file = "/Users/bvossen/Documents/STORM_PALM/analyzed(John)/log_phase/txt_files/SMLM_20190617_dsm_storm_blok_002_20190617_dsm_palm_blok_002.txt"
        file_pairs = [(segmentation_file, points_file)]
        output_dir = '/Users/bvossen/Documents/GitHub/STORMPALM_segmenting/output'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        white_light_file = '/Users/bvossen/Documents/STORM_PALM/analyzed(John)/log_phase/white_light/20190617_dsm_white_002.tif'
    else:
        file_pairs = []  # List to store (segmentation_file, points_file) pairs

        select_white_light_input = input("Do you want to select files for white light? (y/n): ").strip().lower()
        if select_white_light_input == 'y':  
            select_white_light = True
        else:
            select_white_light = False
            white_light_file = None

        # Loop to select multiple files
        # This loop will continue until the user decides to stop adding files
        while True:
            print("Select files for a new batch:")

            print("Select Segmentation file:")
            # Select segmentation file
            segmentation_file = select_file("segmentation file", "*.npy")
            print(f"Selected segmentation file: {segmentation_file}")

            print("Select Points file:")
            # Select points file
            points_file = select_file("points file", "*.txt")
            print(f"Selected points file: {points_file}")

            if select_white_light:
                # Select white light file
                print("Select White Light file:")
                white_light_file = select_file("white light file", ["*.npy", "*.tif", "*.jpg", "*.png"])
                print(f"Selected white light file: {white_light_file}")

            # Add the file pair to the list
            file_pairs.append((segmentation_file, points_file))

            # Ask if the user wants to add more files
            more_files = input("Do you want to add more files? (y/n): ").strip().lower()
            if more_files != 'y':
                break

        # Step 2: Ask for output directory
        print("Select the output directory:")
        output_dir = select_directory()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Process all batches
    for batch_number, (segmentation_file, points_file) in enumerate(file_pairs, start=1):

        # Extract the filename of the points file (without extension)
        points_filename = Path(points_file).stem

        # Create a subdirectory for the current batch using the points filename
        batch_output_dir = output_dir / f"batch_{batch_number}_{points_filename}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        # Load segmentation data
        segmentation_data = load_segmentation_data(segmentation_file)
        masks = segmentation_data.get('masks', None)

        if masks is None:
            raise ValueError(f"No masks found in the segmentation file: {segmentation_file}")

        # Read points from the .txt file
        points = read_points_from_txt(points_file)

        # Scale points from nanometers to pixels (assuming 1 pixel = 100 nm)
        pixel_to_nm = 100
        # points_scaled = points / pixel_to_nm

        # Scale only the x and y coordinates (first two columns), leave the channel (third column) unchanged
        points_scaled = np.column_stack((points[:, :2] / pixel_to_nm, points[:, 2]))

        # Visualize masks and overlay points
        plot_masks(segmentation_file, points_file, batch_output_dir, white_light_file = white_light_file)
        print(f"Processing batch {batch_number}...")

        # Process each mask and save points associated with it
        for mask_id in range(1, masks.max() + 1):  # Assuming mask IDs start from 1
            mask = masks == mask_id

            # Find points that fall within the current mask
            points_in_mask = []
            for point in points_scaled:
                pixel_x, pixel_y, channel = point.astype(int)
                if 0 <= pixel_x < mask.shape[1] and 0 <= pixel_y < mask.shape[0]:
                    if mask[pixel_y, pixel_x]:  # Note: y corresponds to rows, x to columns
                        points_in_mask.append(point)

            points_in_mask = np.array(points_in_mask)

            # Save points associated with the current mask
            if len(points_in_mask) > 0:
                points_output_file = batch_output_dir / f"mask_{mask_id}_points.npy"
                np.save(points_output_file, points_in_mask)
                print(f"Saved points for mask {mask_id} to {points_output_file}")
            else:
                print(f"No points found for mask {mask_id}")

        print(f"Batch {batch_number} processing complete. Outputs saved to {batch_output_dir}")

    print("All batches processed. Exiting...")

if __name__ == "__main__":
    main()