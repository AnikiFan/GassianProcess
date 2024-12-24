import os
import sys
from .download_data import download_data
from .utils import test_1D, dist_spuare

# Set the default working directory to the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
