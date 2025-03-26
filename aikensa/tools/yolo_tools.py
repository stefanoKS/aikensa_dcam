from ultralytics import YOLO

def remove_imageborder_yolo(yolo_results, width=10):
    """
    Adjusts YOLO segmentation results by removing the effect of an added image border.
    
    This function assumes that each result in yolo_results has a `masks` attribute containing:
      - `xy`: a list of numpy arrays with polygon coordinates in absolute pixel space.
      - `xyn`: a list of numpy arrays with normalized polygon coordinates (relative to the padded image).
      - `data`: a numpy array (num_objects x H x W) from which the padded image dimensions can be derived.
    
    For the absolute coordinates (xy), this function subtracts the border offset from each coordinate.
    For the normalized coordinates (xyn), it:
      1. Converts the normalized coordinates to absolute pixel values using the padded image dimensions.
      2. Subtracts the border offset.
      3. Re-normalizes the coordinates relative to the original image dimensions.

    Instead of reassigning the properties (which are read-only), the polygon arrays are modified in place.

    Parameters:
        yolo_results (list): List of YOLO segmentation result objects.
        width (int): Border width in pixels to remove.

    Returns:
        list: Updated YOLO segmentation results with adjusted mask coordinates.
    """
    for result in yolo_results:
        if hasattr(result, 'masks') and result.masks is not None:
            # Adjust absolute polygon coordinates (xy) in place
            if hasattr(result.masks, 'xy') and result.masks.xy is not None:
                for polygon in result.masks.xy:
                    polygon[:, 0] -= width
                    polygon[:, 1] -= width

            # Adjust normalized polygon coordinates (xyn) in place
            if (hasattr(result.masks, 'xyn') and result.masks.xyn is not None and
                    hasattr(result.masks, 'data') and result.masks.data is not None):
                # Derive padded image dimensions from masks.data (shape: num_objects x H x W)
                H, W = result.masks.data.shape[1:3]
                orig_W = W - 2 * width
                orig_H = H - 2 * width
                for polygon in result.masks.xyn:
                    # Work on a copy to compute the new coordinates
                    new_polygon = polygon.copy()
                    # Convert normalized coordinates (relative to padded image) to absolute pixel values
                    new_polygon[:, 0] = new_polygon[:, 0] * W
                    new_polygon[:, 1] = new_polygon[:, 1] * H
                    # Remove the border offset
                    new_polygon[:, 0] -= width
                    new_polygon[:, 1] -= width
                    # Re-normalize coordinates using the original image dimensions
                    new_polygon[:, 0] = new_polygon[:, 0] / orig_W
                    new_polygon[:, 1] = new_polygon[:, 1] / orig_H
                    # Update the original polygon array in place
                    polygon[:] = new_polygon
    return yolo_results