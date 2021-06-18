import numpy

from scipy import ndimage
from scipy.ndimage.filters import convolve

from matplotlib import pyplot
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow

import cv2

# reference : https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123


def gaussian_kernel(size, sigma=1):
	size = int(size) // 2
	x, y = numpy.mgrid[-size: size + 1, -size: size + 1]
	normal = 1 / (2.0 * numpy.pi * sigma ** 2)
	g = numpy.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
	return g


def sobel_filters(img):
	Kx = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], numpy.float32)
	Ky = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], numpy.float32)

	Ix = ndimage.filters.convolve(img, Kx)
	Iy = ndimage.filters.convolve(img, Ky)

	G = numpy.hypot(Ix, Iy)
	G = G / G.max() * 255
	theta = numpy.arctan2(Iy, Ix)

	return (G, theta)


def non_max_suppression(img, D):
	M, N = img.shape
	Z = numpy.zeros((M, N), dtype=numpy.int32)
	angle = D * 180. / numpy.pi
	angle[angle < 0] += 180

	for i in range(1, M-1):
		for j in range(1, N-1):
			try:
				q = 255
				r = 255

			   # angle 0
				if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
					q = img[i, j+1]
					r = img[i, j-1]
				# angle 45
				elif (22.5 <= angle[i, j] < 67.5):
					q = img[i+1, j-1]
					r = img[i-1, j+1]
				# angle 90
				elif (67.5 <= angle[i, j] < 112.5):
					q = img[i+1, j]
					r = img[i-1, j]
				# angle 135
				elif (112.5 <= angle[i, j] < 157.5):
					q = img[i-1, j-1]
					r = img[i+1, j+1]

				if (img[i, j] >= q) and (img[i, j] >= r):
					Z[i, j] = img[i, j]
				else:
					Z[i, j] = 0

			except IndexError as e:
				pass

	return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):

	highThreshold = img.max() * highThresholdRatio
	lowThreshold = highThreshold * lowThresholdRatio

	M, N = img.shape
	res = numpy.zeros((M, N), dtype=numpy.int32)

	weak = numpy.int32(25)
	strong = numpy.int32(255)

	strong_i, strong_j = numpy.where(img >= highThreshold)
	zeros_i, zeros_j = numpy.where(img < lowThreshold)

	weak_i, weak_j = numpy.where(
		(img <= highThreshold) & (img >= lowThreshold))

	res[strong_i, strong_j] = strong
	res[weak_i, weak_j] = weak

	return (res, weak, strong)


def hysteresis(img, weak, strong=255):
	M, N = img.shape
	for i in range(1, M-1):
		for j in range(1, N-1):
			if (img[i, j] == weak):
				try:
					if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
						or (img[i, j-1] == strong) or (img[i, j+1] == strong)
							or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
						img[i, j] = strong
					else:
						img[i, j] = 0
				except IndexError as e:
					pass
	return img


def extract_rgb_color_channel(image_data, channel_index):
	shape = numpy.shape(image_data)
	result = numpy.zeros(shape)

	for x in range(shape[0]):
		for y in range(shape[1]):
			result[x][y][channel_index] = image_data[x][y][channel_index] / 255

	return result

def extract_cmy_color_channel(image_data, channel_index):
	shape = numpy.shape(image_data)
	result = numpy.zeros(shape)

	for x in range(shape[0]):
		for y in range(shape[1]):
			result[x][y][0] = 1
			result[x][y][1] = 1
			result[x][y][2] = 1

			result[x][y][channel_index] = (
				255 - image_data[x][y][channel_index]) / 255
	return result

def rgb_to_bw(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
	return gray

def plot_image_with_rgb_channels(image_path):
	image_data = imread(image_path)

	fig, axes = pyplot.subplots(2, 2)
	fig.suptitle("Red Green Blue")
	axes[0][0].set_title("Original")
	axes[0][0].imshow(image_data)
	axes[0][1].set_title("Red")
	axes[0][1].imshow(extract_rgb_color_channel(image_data, 0))
	axes[1][0].set_title("Green")
	axes[1][0].imshow(extract_rgb_color_channel(image_data, 1))
	axes[1][1].set_title("Blue")
	axes[1][1].imshow(extract_rgb_color_channel(image_data, 2))

	pyplot.show()


def plot_image_with_cmy_channels(image_path):
	image_data = imread(image_path)

	fig, axes = pyplot.subplots(2, 2)
	fig.suptitle("Cyan Magenta Yellow")
	axes[0][0].set_title("Original")
	axes[0][0].imshow(image_data)
	axes[0][1].set_title("Cyan")
	axes[0][1].imshow(extract_cmy_color_channel(image_data, 0))
	axes[1][0].set_title("Magenta")
	axes[1][0].imshow(extract_cmy_color_channel(image_data, 1))
	axes[1][1].set_title("Yellow")
	axes[1][1].imshow(extract_cmy_color_channel(image_data, 2))

	pyplot.show()


def plot_image_black_and_white(image_path):
	image_data = imread(image_path)

	fig, axes = pyplot.subplots(2, 1)
	fig.suptitle("Black and white")
	axes[0].set_title("Original")
	axes[0].imshow(image_data)
	axes[1].set_title("Black and White")
	axes[1].imshow(rgb_to_bw(image_data), 'gray')

	pyplot.show()

def plot_canny_filter(image_path):
	image_data = imread(image_path)
	image_bw = rgb_to_bw(image_data)

	# appliquer un filtre gaussien sur l'image en noir et blanc afin d'éliminer le bruit
	smooth_kernel = gaussian_kernel(size=5, sigma=1.4)
	smooth_image = convolve(image_bw, smooth_kernel)

	# appliquer un filtre de sobel
	gradient, theta = sobel_filters(smooth_image)

	non_max_image = non_max_suppression(gradient, theta)

	(threshold_image, weak, strong) = threshold(non_max_image)

	our_image_border = hysteresis(threshold_image, weak, strong)

	opencv_image_border = cv2.Canny(image_data, 200, 100)

	fig, axes = pyplot.subplots(3, 3)
	fig.suptitle("Steps of Canny filter")
	axes[0][0].set_title("Original BW Image")
	axes[0][0].imshow(image_bw, 'gray')
	axes[0][1].set_title("Noise Reduction")
	axes[0][1].imshow(smooth_image, 'gray')
	axes[0][2].set_title("Gradient Calculation")
	axes[0][2].imshow(gradient, 'gray')
	axes[1][0].set_title("Non-Maximum Suppression")
	axes[1][0].imshow(non_max_image, 'gray')
	axes[1][1].set_title("Double threshold")
	axes[1][1].imshow(threshold_image, 'gray')
	axes[1][2].set_title("Edge Tracking by Hysteresis")
	axes[1][2].imshow(our_image_border, 'gray')
	axes[2][0].set_title("Canny openCV")
	axes[2][0].imshow(opencv_image_border, 'gray')

	# 2 2 notre fft
	image_fft = numpy.fft.fft2(our_image_border)
	image_fft_shifted = numpy.fft.fftshift(image_fft)
	fft_magnitude_spectrum = 20 * numpy.log(numpy.abs(image_fft_shifted))
	
	axes[2][2].set_title("FFT on Canny edge detector")
	axes[2][2].imshow(fft_magnitude_spectrum, 'gray')

	# 2 1 fft de l'image d'origine
	bw_fft = numpy.fft.fft2(image_bw)
	bw_fft_shifted = numpy.fft.fftshift(bw_fft)
	bw_magnitude_spectrum = 20 * numpy.log(numpy.abs(bw_fft_shifted))
	
	axes[2][1].set_title("FFT on Black and White image")
	axes[2][1].imshow(bw_magnitude_spectrum, 'gray')

	pyplot.show()


if __name__ == '__main__':
	image_path = "chat_qui_mange_des_sushis.jpg"

	plot_image_with_rgb_channels(image_path)
	plot_image_with_cmy_channels(image_path)
	plot_image_black_and_white(image_path)
	plot_canny_filter(image_path)
