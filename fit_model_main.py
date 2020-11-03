from utils.visualization_helpers import load_images, show_images
from models.DeepDecoder import DeepDecoder
from models import ConvolutionDecoder
from fitting.Fitter import Fitter

IMAGE_PATHS = ["data/raw_images/canonical_noisy.png", "data/raw_images/canonical_target.png"]
MODEL_TYPE = "deep" # "conv": ConvolutionDecoder; "deep": DeepDecoder
INPUT_SHAPE = [8, 8]  # Deep Decoder 8,8, Conv Decoder 2,4
NUMBER_OF_HIDDEN_LAYERS = 6  # Deep Decoder 6, ConvDecoder 8
NUMBER_OF_HIDDEN_CHANNELS = 64  # Deep Decoder 128, ConvDecoder 64

if __name__ == '__main__':
    gibbs_image, target_image = load_images(IMAGE_PATHS)
    if MODEL_TYPE == "conv":
        model = ConvolutionDecoder(INPUT_SHAPE, gibbs_image.shape, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_HIDDEN_CHANNELS)
    elif MODEL_TYPE == "deep":
        model = DeepDecoder(INPUT_SHAPE, gibbs_image.shape, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_HIDDEN_CHANNELS)
    fitter = Fitter(30000, convergence_check_length=100, find_best=True, log_frequency=1)
    fitter(model, gibbs_image, target_image)
    model_image = fitter.get_best_image()
    show_images(gibbs_image, model_image, target_image, str(model), save_plot=True)
