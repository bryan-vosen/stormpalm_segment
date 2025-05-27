def load_segmentation_data(segmentation_file):
    """
    Load segmentation data from a .npy file.

    Parameters:
    segmentation_file (str): Path to the segmentation .npy file.

    Returns:
    dict: Segmentation data containing masks and other metadata.
    """
    import numpy as np
    return np.load(segmentation_file, allow_pickle=True).item()

def read_points_from_txt(txt_file):
    """
    Read points from a .txt file and return them as a NumPy array.

    Parameters:
    txt_file (str): Path to the .txt file containing points.

    Returns:
    np.ndarray: Array of points with columns for x and y positions and channel.
    """
    import pandas as pd
    import numpy as np

    try:
        points_df = pd.read_csv(txt_file, delimiter='\t')
        if 'position_x [nm]' in points_df.columns and 'position_y [nm]' in points_df.columns:
            points = np.asarray(points_df[['position_x [nm]', 'position_y [nm]', 'channel']])
        else:
            points = np.asarray(points_df[['position_x', 'position_y', 'channel']])
    except Exception as e:
        raise ValueError(f"Error reading points from file: {e}")

    return points