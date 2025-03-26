import cv2

def add_imageborder(img, color=(255, 255, 255), width=10):
    """
    Adds a border to the image.

    Parameters:
        img (numpy.ndarray): Input image.
        color (tuple): Border color in BGR format (default is white, i.e. (255, 255, 255)).
        width (int): Width of the border in pixels.

    Returns:
        numpy.ndarray: Image with the added border.
    """
    bordered_img = cv2.copyMakeBorder(img, width, width, width, width, 
                                      borderType=cv2.BORDER_CONSTANT, value=color)
    return bordered_img

    