"""Tests for the Visualizer class."""



import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from hamacho.core.post_processing.visualizer import Visualizer


def test_visualize_fully_defected_masks():
    """Test if a fully defected anomaly mask results in a completely white image."""

    # create visualizer and add fully defected mask
    visualizer = Visualizer(save_combined_result_image=True)
    mask = np.ones((256, 256)) * 255
    visualizer.add_image(image=mask, color_map="gray", title="fully defected mask")
    visualizer.generate(image_name="unit-test-code.jpg")

    # retrieve plotted image
    canvas = FigureCanvas(visualizer.figure)
    canvas.draw()
    plotted_img = visualizer.axis.images[0].make_image(canvas.renderer)

    # assert that the plotted image is completely white
    assert np.all(plotted_img[0][..., 0] == 255)
