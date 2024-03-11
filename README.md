![Python](https://img.shields.io/badge/python-3670A0?style=plastic&logo=python&logoColor=ffdd54)

# [JPEG Compression Demostration](https://github.com/PRO-GUNE/JPEGCompression.git)
This repository demonstrates, using an example numpy array, how the JPEG Compression process works.

# How to Use the Project
To use the project you need to have Python version above or equal to 3.9.6 installed

1. Clone the project
   `git clone <repository url>`
2. Open the cloned project folder in the terminal. Move to the `src` directory and create a virtual environment and install the required dependencies
   `cd ./src
    python -m venv .
    pip install -r requirements.txt`
3. Run the main python file
   `python functions.py`

## Convert image to YCrCb format
The RGB color space is converted to YCbCr to separate luminance (Y) and chrominance (Cb and Cr) components. This conversion exploits the human eye's higher sensitivity to brightness (luminance) compared to color information.

_Importance: Separating color information allows for more efficient compression, as chrominance components can be subsampled without significant loss in image quality_

## Sample image into 8x8 blocks
The image is divided into non-overlapping 8x8 blocks for further processing. Sampling ensures that each 8x8 block represents a portion of the image, allowing for localized transformations.

_Importance: Working with small blocks simplifies the compression process. Each block can be processed independently, enabling parallelization and facilitating various compression techniques._

## Perform Discrete Cosine Transform(DCT)
DCT is applied to each 8x8 block, transforming spatial domain pixel values into frequency domain coefficients. DCT helps concentrate most of the image information into a few low-frequency coefficients, making 
subsequent quantization and compression more effective.

_Importance: DCT reduces spatial redundancy and prepares the data for quantization, making it suitable for lossy compression while minimizing perceptual loss_

## Quantization
Quantization involves dividing DCT coefficients by a quantization matrix. This step introduces loss by mapping a wide range of values to a limited set. Higher frequencies, which are less perceptible to the human eye, are quantized more heavily, contributing to data reduction.

_Importance: Quantization is a key lossy step, enabling significant data reduction. The choice of quantization matrix balances compression ratio and image quality_

## Encoding
Entropy encoding is a data compression technique used to represent data in the most efficient way possible, reducing the amount of data needed to convey information.
In JPEG Compression, Entropy encoding is done in 3 steps

### ZigZag reordering of DCT coefficients
Zigzag scanning reorganizes quantized DCT coefficients into a linear array. This process groups coefficients with similar frequencies together, increasing the efficiency of run-length encoding and entropy coding

_Importance: Zigzag scanning prepares the data for run-length encoding, ensuring that non-zero coefficients are efficiently represented._

### Apply DPCM encoding for the DC Component and RLE encoding for the AC Components
DPCM captures the relative changes in brightness for the DC component, ensuring efficient representation and lossless compression. RLE targets sequences of zeros in the AC components, enabling the compression of sparse data structures and eliminating redundancy.

_Importance: These techniques enhance storage, transmission, and processing efficiency while preserving the original data quality, making them fundamental components of many image compression algorithms._

###  Huffman Encoding
Huffman encoding is a fundamental technique in data compression, leveraging the frequency distribution of characters to generate efficient variable-length codes. 

_Importance: This method achieves compact data representation, enabling significant reduction in storage space and transmission bandwidth_

## Conclusion
This is a demonstration of how the JPEG Compression on one channel of a image that is sampled into 8x8 blocks. This algorithm is performed on all the channels of the image and finally the entire image will be encoded.

