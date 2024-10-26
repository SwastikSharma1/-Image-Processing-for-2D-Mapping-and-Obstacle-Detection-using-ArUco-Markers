# Image-Processing-for-2D-Mapping-and-Obstacle-Detection-using-ArUco-Markers

## Overview

The Image-Processing-for-2D-Mapping-and-Obstacle-Detection-using-ArUco-Markers project utilizes OpenCV and ArUco markers for detecting obstacles in images and calculating their real-world area. 
This Python application processes images, identifies ArUco markers, and detects obstacles within the defined area marked by the ArUco markers.

## Features

- Detects ArUco markers using predefined dictionaries.
- Identifies obstacles in the image using contour detection.
- Applies a perspective transformation based on detected markers.
- Calculates the total area of detected obstacles and estimates real-world area using a predefined scale.
- Displays processed images with detected contours and ArUco markers.

## Requirements

- Python 3.11.2
- OpenCV
- NumPy

You can install the required libraries using pip:

```bash
pip install opencv-python numpy
