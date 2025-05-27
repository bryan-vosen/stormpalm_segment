# Segmentation Mask Processor

This project is designed to process segmentation masks from `.npy` files and associated point data from `.txt` files. It extracts individual mask segments, saves them as separate numpy arrays, and visualizes their locations on the original image.

## Project Structure

```
segmentation-mask-processor
├── src
│   ├── main.py                # Entry point of the script
│   ├── utils
│   │   └── file_operations.py  # Utility functions for file operations
│   └── visualization
│       └── plot_masks.py      # Functions for visualizing masks
├── output                      # Directory for generated files
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your segmentation `.npy` file and the corresponding `.txt` file in the appropriate directory.
2. Run the main script:

```bash
python src/main.py
```

3. The processed mask segments will be saved as numpy arrays in the `output` directory, along with a visualization image showing the locations of each mask.

## Dependencies

This project requires the following Python packages:

- numpy
- matplotlib

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License.