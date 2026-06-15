
import numpy as np
import cmapy


EDGE_TYPE_COLORS = {
    'q': "#FF00FF", # Purple
    'k': "#00FF00", # Green
    'v': "#0000FF", # Blue
    None: "#000000", # Black
}

def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """

    def rgb2hex(rgb):
        """
        https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexadecimal-string
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    return rgb2hex(cmapy.color(colorscheme, np.random.randint(0, 256), rgb_order=True))


def model2family(model_name: str):
    if 'gpt2' in model_name:
        return 'gpt2'
    elif 'pythia' in model_name:
        return 'pythia'
    elif model_name.startswith(('csp_', 'dense1_')) or 'circuit-sparsity' in model_name:
        return 'tinypython_2k'
    else:
        raise ValueError(f"Couldn't find model family for model: {model_name}")
