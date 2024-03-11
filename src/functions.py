import numpy as np
import math


# print function
def print_with_tab(mat):
    m, n = mat.shape
    for i in range(m):
        for j in range(n):
            print(round(mat[i][j]), end="\t")
        print()


# Convert RGB to YCbCr
def RGB2YCbCr(image: np.ndarray) -> np.ndarray:
    # Convert RGB to YCrCb
    offset = np.array([16, 128, 128])
    ycbcr_transform = np.array(
        [
            [0.257, 0.504, 0.098],
            [-0.148, -0.291, 0.439],
            [0.439, -0.368, -0.071],
        ]
    )

    ycbcr_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ycbcr_image[i, j] = np.round(np.dot(ycbcr_transform, image[i, j]) + offset)

    return ycbcr_image


# DCT function
def dct(image_block):
    dct_block = np.zeros_like(image_block, dtype=np.float32)
    for u in range(8):
        for v in range(8):
            cu = 1 if u == 0 else np.sqrt(2)
            cv = 1 if v == 0 else np.sqrt(2)
            sum_val = 0
            for x in range(8):
                for y in range(8):
                    sum_val += (
                        image_block[x, y]
                        * np.cos((2 * x + 1) * u * np.pi / 16)
                        * np.cos((2 * y + 1) * v * np.pi / 16)
                    )
            dct_block[u, v] = 0.25 * cu * cv * sum_val

    return dct_block


# Quantizing constants
luminance_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.uint16,
)

chrominance_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.uint16,
)


def quantize_block(block, quantization_matrix):
    return np.round(block / quantization_matrix)


# Quantization function
def quantize(dct_block):
    quantized_luminance_block = quantize_block(
        dct_block[..., 0], luminance_quantization_matrix
    )
    quantized_chrominance_blocks = [
        quantize_block(dct_block[..., i], chrominance_quantization_matrix)
        for i in range(1, 3)
    ]

    return quantized_luminance_block, quantized_chrominance_blocks


# zigzag scan
def zigzag_scan(block):
    zigzag = []
    for i in range(15):
        if i < 8:
            for j in range(i + 1):
                if i % 2 == 0:
                    zigzag.append(block[i - j][j])
                else:
                    zigzag.append(block[j][i - j])
        else:
            for j in range(15 - i):
                if i % 2 == 0:
                    zigzag.append(block[7 - j][j + (i - 7)])
                else:
                    zigzag.append(block[j + (i - 7)][7 - j])

    return np.array(zigzag, dtype=np.int32)


# Entropy Encoding
# DPCM encoding for DC component
def dpcm_encode_dc(dc_coefficients):
    dpcm_encoded_dc = [dc_coefficients[0]]  # Initialize with the first DC coefficient
    for i in range(1, len(dc_coefficients)):
        # Encode the difference between current DC coefficient and previous DC coefficient
        diff = dc_coefficients[i] - dc_coefficients[i - 1]
        dpcm_encoded_dc.append(diff)
    return dpcm_encoded_dc


# RLE for AC components
def run_length_encode_ac(ac_coefficients):
    rle_encoded_ac = []
    run_length = 1
    prev_chr = ac_coefficients[0]
    for ac_coefficient in ac_coefficients[1:]:
        if ac_coefficient == prev_chr:
            run_length += 1
        else:
            # Append (run length, value) pair to the encoded list
            rle_encoded_ac.extend([prev_chr, run_length])
            prev_chr = ac_coefficient
            run_length = 1

    # Add the last entry
    rle_encoded_ac.extend([prev_chr, run_length])
    return rle_encoded_ac


import heapq
from collections import defaultdict


# Step 1: Find frequency dictionary
def build_frequency_dict(data):
    frequency_dict = defaultdict(int)
    for char in data:
        frequency_dict[char] += 1
    return frequency_dict


# Step 2: Huffman algorithm
def build_huffman_tree(frequency_dict):
    priority_queue = [[weight, [char, ""]] for char, weight in frequency_dict.items()]
    heapq.heapify(priority_queue)
    while len(priority_queue) > 1:
        lo = heapq.heappop(priority_queue)
        hi = heapq.heappop(priority_queue)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(priority_queue, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return priority_queue[0]


# Step 3: Generate Huffman codes
def generate_huffman_codes(huffman_tree):
    huffman_codes = {}
    for pair in huffman_tree[1:]:
        char = pair[0]
        code = pair[1]
        huffman_codes[char] = code
    return huffman_codes


# Huffman encoding
def huffman_encode(rle, codes):
    encoded_str = ""
    for sym in rle:
        encoded_str += codes[sym]

    return encoded_str


def main():
    # Create an 8x8 NumPy array
    image_block = np.array(
        [
            [52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 55, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87, 79, 69, 68, 65, 76, 78, 94],
        ],
        dtype=np.uint8,
    )
    print("Image: ")
    print_with_tab(image_block)

    # perform DCT
    dct_block = dct(image_block)
    print("DCT block:")
    print_with_tab(dct_block)

    # perform quantization
    quantized_luminance_block, quantized_chrominance_blocks = quantize(dct_block)
    print("Quantized Luminance block")
    print_with_tab(quantized_luminance_block)

    zigzag_luminance_block = zigzag_scan(quantized_luminance_block)
    print("ZigZag scan")
    print(zigzag_luminance_block)

    dpcm_encoded_dc_val = dpcm_encode_dc(zigzag_luminance_block[:1])
    rle_encoded_ac = run_length_encode_ac(zigzag_luminance_block[1:])

    encoded_val = np.array(dpcm_encoded_dc_val + rle_encoded_ac)
    print("Encoded values: ")
    print(encoded_val)

    freq_dict = build_frequency_dict(encoded_val)
    print("Frequency dictionary: ")
    print(dict(freq_dict))

    huffman_tree = build_huffman_tree(freq_dict)
    print("Huffman Tree: ")
    print(huffman_tree)

    huffman_codes = generate_huffman_codes(huffman_tree)
    print("Huffman Codes: ")
    print(huffman_codes)

    huffman_encoded_str = huffman_encode(encoded_val, huffman_codes)
    print("Huffman Encoded String: ")
    print(huffman_encoded_str)


main()
