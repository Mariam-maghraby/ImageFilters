import re
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from cv2 import equalizeHist
from guiWithoutGraphics import Ui_MainWindow
import pyqtgraph as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import cv2


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.Browse_Button.clicked.connect(self.open_file)
        self.ui.Choose_Filter.activated.connect(self.choose_filter)
        self.ui.Histogram_Button.clicked.connect(self.histogram_view)

        self.input_img_canvas = MplCanvas2(self)
        self.input_img_freq_canvas = MplCanvas2(self)
        self.filtered_img_canvas = MplCanvas2(self)
        self.filtered_img_freq_canvas = MplCanvas2(self)
        self.original_histogram_canvas = MplCanvas2(self)
        self.equalized_histogram_canvas = MplCanvas2(self)
        self.input_img_hist_canvas = MplCanvas2(self)
        self.equalized_img_canvas = MplCanvas2(self)

        self.fig_size = 9
        self.saved_img = 'filtered Image.png'

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(
            None, "QFileDialog.getOpenFileName()", "", "All Files (*);;csv Files (*.csv)", options=options)
        if self.fileName:
            self.read_file(self.fileName)
            self.freq_domain(self.fileName)
            self.input_img_freq_canvas.axes.imshow(
                self.magnitude_spectrum, cmap='gray')
            self.input_img_freq_canvas.draw()
            self.ui.verticalLayout_3.addWidget(self.input_img_freq_canvas)

    def read_file(self, file_path):
        path = file_path
        im = cv2.imread(path)
        if len(im.shape) < 2:
            self.input_img_canvas.axes.imshow(im)
            self.input_img_canvas.draw()
            self.ui.verticalLayout_4.addWidget(self.input_img_canvas)
        elif len(im.shape) == 3:
            self.input_img_canvas.axes.imshow(
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            self.input_img_canvas.draw()
            self.ui.verticalLayout_4.addWidget(self.input_img_canvas)

    def freq_domain(self, file_path):
        img = cv2.imread(file_path, 0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        self.magnitude_spectrum = 20 * \
            np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    def choose_filter(self):
        self.index = self.ui.Choose_Filter.currentIndex()
        if self.index == 1:
            self.Mean_Filter(self.fileName)
        elif self.index == 2:
            self.Gaussian_Filter(self.fileName)
        # elif index == 3:
        #    self.High_Pass_Filter(self.fileName)
        else:
            self.Frequency_Filter(self.fileName)

    def Mean_Filter(self, file_path):
        original_img = cv2.imread(file_path)
        if len(original_img.shape) == 3:
            converted_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            filtered_img = cv2.blur(
                converted_img, (self.fig_size, self.fig_size))
            result_img = cv2.cvtColor(filtered_img, cv2.COLOR_HSV2RGB)
            self.filtered_img_canvas.axes.imshow(result_img)
            self.filtered_img_canvas.draw()
            cv2.imwrite(self.saved_img, result_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()

        else:
            filtered_img = cv2.blur(
                original_img, (self.fig_size, self.fig_size))
            self.filtered_img_canvas.axes.imshow(
                filtered_img, cmap='gray')
            self.filtered_img_canvas.draw()
            self.ui.verticalLayout_5.addWidget(self.filtered_img_canvas)
            cv2.imwrite(self.saved_img, filtered_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()

    def Gaussian_Filter(self, file_path):
        original_img = cv2.imread(file_path)
        if len(original_img.shape) == 3:
            converted_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            filtered_img = cv2.GaussianBlur(
                converted_img, (self.fig_size, self.fig_size), 0)
            result_img = cv2.cvtColor(filtered_img, cv2.COLOR_HSV2RGB)
            self.filtered_img_canvas.axes.imshow(result_img)
            self.filtered_img_canvas.draw()
            self.ui.verticalLayout_5.addWidget(self.filtered_img_canvas)
            cv2.imwrite(self.saved_img, result_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()
        else:
            filtered_img = cv2.GaussianBlur(
                original_img, (self.fig_size, self.fig_size), 0)
            self.filtered_img_canvas.axes.imshow(
                filtered_img, cmap='gray')
            self.filtered_img_canvas.draw()
            self.ui.verticalLayout_5.addWidget(self.filtered_img_canvas)
            cv2.imwrite(self.saved_img, filtered_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()

    def Frequency_Filter(self, file_path):
        original_img = cv2.imread(file_path)
        if len(original_img.shape) == 3:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img_dft = np.fft.fft2(original_img)
        dft_shift = np.fft.fftshift(img_dft)
        if self.index == 3:
            dft_shift = self.highPassFiltering(dft_shift, 100)
        elif self.index == 4:
            dft_shift = self.lowPassFiltering(dft_shift, 50)

        # Move the frequency domain from the middle to the upper left corner
        idft_shift = np.fft.ifftshift(dft_shift)
        ifimg = np.fft.ifft2(idft_shift)  # Fourier library function call
        ifimg = np.abs(ifimg)
        self.filtered_img_freq_canvas.axes.imshow(ifimg, cmap='gray')
        self.filtered_img_freq_canvas.draw()
        self.ui.verticalLayout_6.addWidget(self.filtered_img_freq_canvas)
        img_dft = np.fft.fft2(ifimg)
        dft_shift = np.fft.fftshift(img_dft)
        # abs is equivalent to Norm-2 L2
        magnitude_spectrum = np.abs(dft_shift)
        magnitude_spectrum_log = np.log(magnitude_spectrum+1)
        self.input_img_freq_canvas.axes.imshow(
            magnitude_spectrum_log, cmap='gray')
        self.input_img_freq_canvas.draw()
        self.ui.verticalLayout_3.addWidget(self.input_img_freq_canvas)

    # Transfer parameters are Fourier transform spectrogram and filter size
    def highPassFiltering(self, img, size):
        h, w = img.shape[0:2]  # Getting image properties
        # Find the center point of the Fourier spectrum
        h1, w1 = int(h/2), int(w/2)
        # Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
        img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0
        return img

    # Transfer parameters are Fourier transform spectrogram and filter size
    def lowPassFiltering(self, img, size):
        h, w = img.shape[0:2]  # Getting image properties
        # Find the center point of the Fourier spectrum
        h1, w1 = int(h/2), int(w/2)
        # Define a blank black image with the same size as the Fourier Transform Transfer
        img2 = np.zeros((h, w), np.uint8)
        # Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
        img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
        # A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.
        img3 = img2*img
        return img3

    def draw_freq_filtered_Img(self):
        self.filtered_img_freq_canvas.axes.imshow(
            self.magnitude_spectrum, cmap='gray')
        self.filtered_img_freq_canvas.draw()
        self.ui.verticalLayout_6.addWidget(self.filtered_img_freq_canvas)

    def read_input_image(self):
        img = cv2.imread(self.fileName, 0)
        return img

    def histogram_view(self):
        # hide the filters layout
        # show the histogram layout
        self.equalized_img()
        self.plot_original_histogram()
        self.plot_equalized_histogram()
        self.ui.verticalLayout_8.addWidget(self.input_img_canvas)

    def flaten_image(self):
        img = self.read_input_image()
        img = np.asarray(img)
        flat_img = img.flatten()  # flats the matrix of img into 1D array
        #self.original_histogram_canvas.plt.hist(flat_img, bins=50) ####
        # print(flat_img)
        return flat_img

    def original_histogram(self, bins):
        img = self.flaten_image()
        histogram = np.zeros(bins)
        for pixel in img:
            histogram[pixel] += 1
        # print(histogram)
        return histogram

    def plot_original_histogram(self):
        hist = self.original_histogram(256)
        self.original_histogram_canvas.axes.cla()
        self.original_histogram_canvas.axes.plot(hist)
        self.ui.verticalLayout_12.addWidget(self.original_histogram_canvas)

    def cummulative_sum(self, a):
        a = iter(a)
        b = [next(a)]

        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def equalized_histogram(self):
        hist = self.original_histogram(256)
        hist_cumm_sum = self.cummulative_sum(hist)
        nj = (hist_cumm_sum - hist_cumm_sum.min()) * 255
        no_of_bins = hist_cumm_sum.max() - hist_cumm_sum.min()
        # re-normalize the cummulative sum
        hist_cumm_sum = nj / no_of_bins
        # 8 bits integer typecasting to avoid floating point values
        hist_cumm_sum = hist_cumm_sum.astype('uint8')
        return hist_cumm_sum

    def plot_equalized_histogram(self):
        equalized_hist = self.equalized_histogram()
        self.equalized_histogram_canvas.axes.cla()
        self.equalized_histogram_canvas.axes.plot(equalized_hist)
        self.ui.verticalLayout_13.addWidget(self.equalized_histogram_canvas)

    def equalized_img(self):
        equalized_hist = self.equalized_histogram()
        equalized_img = equalized_hist[self.flaten_image()]
        original_img = self.read_input_image()
        equalized_img = np.reshape(equalized_img, original_img.shape)
        self.equalized_img_canvas.axes.clear()
        self.equalized_img_canvas.axes.imshow(equalized_img, cmap='gray')
        self.ui.verticalLayout_9.addWidget(self.equalized_img_canvas)


class MplCanvas2(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=3.5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.tight_layout()
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas2, self).__init__(self.fig)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ApplicationWindow()
    MainWindow.show()
    sys.exit(app.exec_())
