import numpy as np
import pandas as pd
from .functions import *
from .clean_data import *

def add_clms(data):
    data["Time"] = (data["Time"] // 3600) % 24
    return data

def delete(data):
    return data

def remove_outliers(data, mn, mx):
    return data

def clean_clms(data):
    return data