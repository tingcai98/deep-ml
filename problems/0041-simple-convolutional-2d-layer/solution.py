import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# Your code here
	padding = np.pad(
		input_matrix,
		((padding, padding), (padding, padding)),
		mode = "constant"
	)
	H, W = padding.shape

	out_h = (H-kernel_height) // stride + 1
	out_w = (W-kernel_width) // stride + 1

	output = np.zeros((out_h, out_w))

	for i in range(out_h):
		for j in range(out_w):
			h_start = i*stride
			w_start = j*stride

			window = padding[
				h_start:h_start+kernel_height,
				w_start:w_start+kernel_width
			]

			output[i,j] = np.sum(window*kernel)
    
	return output
