import qdarkstyle
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import cv2
from Dark_Mode_App_Filtering import Ui_MainWindow


class MplCanvas2(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=3.5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('#19232d')
        self.fig.tight_layout()
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas2, self).__init__(self.fig)


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.translate = QtCore.QCoreApplication.translate
        self.ui.Browse_Button.clicked.connect(self.open_file)
        self.ui.Choose_Filter.activated.connect(self.choose_filter)
        self.ui.Filter_Button.clicked.connect(self.show_filters)
        self.ui.Histogram_Button.clicked.connect(self.show_histogram)
        self.Canvas_for_input_Image = MplCanvas2(self)
        self.Canvas_for_Frequency_Domain_Image = MplCanvas2(self)
        self.Canvas_for_Filtered_Image_Image = MplCanvas2(self)
        self.Canvas_for_Frequency_Domain_filteredImage = MplCanvas2(self)
        self.Canvas_for_Original_Histogram = MplCanvas2(self)
        self.Canvas_for_Equalized_Histogram = MplCanvas2(self)
        self.Canvas_for_Equalized_Image = MplCanvas2(self)
        self.ui.verticalLayout_4.addWidget(self.Canvas_for_input_Image)
        self.ui.verticalLayout_3.addWidget(
            self.Canvas_for_Frequency_Domain_Image)
        self.ui.verticalLayout_3.addWidget(self.Canvas_for_Original_Histogram)
        self.ui.verticalLayout_5.addWidget(self.Canvas_for_Equalized_Image)
        self.ui.verticalLayout_5.addWidget(
            self.Canvas_for_Filtered_Image_Image)
        self.ui.verticalLayout_6.addWidget(
            self.Canvas_for_Frequency_Domain_filteredImage)
        self.ui.verticalLayout_6.addWidget(self.Canvas_for_Equalized_Histogram)
        self.canvas = [self.Canvas_for_input_Image, self.Canvas_for_Frequency_Domain_Image, self.Canvas_for_Filtered_Image_Image,
                       self.Canvas_for_Frequency_Domain_filteredImage, self.Canvas_for_Original_Histogram, self.Canvas_for_Equalized_Histogram,
                       self.Canvas_for_Equalized_Image, self.ui.Filtered_Image_Title, self.ui.Frequency_Domain_filtered_Title]
        for canva in self.canvas:
            canva.setVisible(False)
        self.ui.splitter.setSizes([800, 200])  # was:(800,400)
        self.ui.splitter_2.setSizes([400, 20])  # was:(350,20)
        self.fig_size = 9
        self.saved_img = 'filtered Image.png'

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(
            None, "QFileDialog.getOpenFileName()", "", "All Files (*);;csv Files (*.csv)", options=options)
        if self.fileName:
            self.show_or_hide_widgets()
            self.Canvas_for_input_Image.setVisible(True)
            self.Canvas_for_Frequency_Domain_Image.setVisible(True)
            self.Canvas_for_Original_Histogram.setVisible(False)
            self.ui.Frequency_Domain_Title.setText(self.translate(
                "MainWindow", "<html><head/><body><p align=\"center\">Frequency Domain</p></body></html>"))
            self.read_file(self.fileName)
            self.freq_domain(self.fileName)
            self.Canvas_for_Frequency_Domain_Image.axes.imshow(self.magnitude_spectrum, cmap='gray', extent=[
                0.5, (self.magnitude_spectrum.shape[1]+0.5), 0.5, (self.magnitude_spectrum.shape[0]+0.5)])
            self.Canvas_for_Frequency_Domain_Image.draw()

    def read_file(self, file_path):
        path = file_path
        self.im = cv2.imread(path)
        print(self.im.shape)
        if len(self.im.shape) < 2:
            self.Canvas_for_input_Image.axes.imshow(self.im, extent=[
                0.5, (self.im.shape[1]), 0.5, (self.im.shape[0]+0.5)])
            self.Canvas_for_input_Image.draw()
        elif len(self.im.shape) == 3:
            self.Canvas_for_input_Image.axes.imshow(cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB), extent=[
                0.5, (self.im.shape[1]), 0.5, (self.im.shape[0]+0.5)])
            self.Canvas_for_input_Image.draw()

    def freq_domain(self, file_path):
        #img = self.im.copy()
        img = cv2.imread(file_path, 0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        self.magnitude_spectrum = 20 * \
            np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    def choose_filter(self):
        self.Canvas_for_Filtered_Image_Image.axes.cla()
        self.Canvas_for_Frequency_Domain_filteredImage.axes.cla()
        self.index = self.ui.Choose_Filter.currentIndex()
        print(f"index:{self.index}")
        filters_combo_box = [self.Mean_Filter, self.Median_Filter,
                             self.Gaussian_Filter, self.TwoD_Filter, self.Box_Filter,
                             self.Laplacian_Filter]

        if(self.index >= 3):
            for i in range(3, 9):
                if self.index == i:
                    print(f"i= {i}")
                    filters_combo_box[i-3]()
        else:
            self.Frequency_Filter()

    def Mean_Filter(self):
        #original_img = cv2.imread(file_path)
        original_img = self.im.copy()
        if len(original_img.shape) == 3:
            converted_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            filtered_img = cv2.blur(
                converted_img, (self.fig_size, self.fig_size))
            result_img = cv2.cvtColor(filtered_img, cv2.COLOR_HSV2RGB)
            self.Canvas_for_Filtered_Image_Image.axes.imshow(result_img, extent=[
                0.5, (result_img.shape[1]), 0.5, (result_img.shape[0]+0.5)])
            self.Canvas_for_Filtered_Image_Image.draw()
            cv2.imwrite(self.saved_img, result_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()
        else:
            filtered_img = cv2.blur(
                original_img, (self.fig_size, self.fig_size))
            self.Canvas_for_Filtered_Image_Image.axes.imshow(filtered_img, cmap='gray', extent=[
                0.5, (filtered_img.shape[1]), 0.5, (filtered_img.shape[0]+0.5)])
            self.Canvas_for_Filtered_Image_Image.draw()
            cv2.imwrite(self.saved_img, filtered_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()

    def Gaussian_Filter(self):
        #original_img = cv2.imread(file_path)
        original_img = self.im.copy()
        if len(original_img.shape) == 3:
            converted_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            filtered_img = cv2.GaussianBlur(
                converted_img, (self.fig_size, self.fig_size), 0)
            result_img = cv2.cvtColor(filtered_img, cv2.COLOR_HSV2RGB)
            self.Canvas_for_Filtered_Image_Image.axes.imshow(result_img, extent=[
                0.5, (result_img.shape[1]), 0.5, (result_img.shape[0]+0.5)])
            self.Canvas_for_Filtered_Image_Image.draw()
            cv2.imwrite(self.saved_img, result_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()
        else:
            filtered_img = cv2.GaussianBlur(
                original_img, (self.fig_size, self.fig_size), 0)
            self.Canvas_for_Filtered_Image_Image.axes.imshow(filtered_img, cmap='gray', extent=[
                0.5, (filtered_img.shape[1]), 0.5, (filtered_img.shape[0]+0.5)])
            self.Canvas_for_Filtered_Image_Image.draw()
            cv2.imwrite(self.saved_img, filtered_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()

    def Median_Filter(self):
        #original_img = cv2.imread(file_path)
        original_img = self.im.copy()
        if len(original_img.shape) == 3:
            converted_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            filtered_img = cv2.medianBlur(converted_img, self.fig_size)
            result_img = cv2.cvtColor(filtered_img, cv2.COLOR_HSV2RGB)
            self.Canvas_for_Filtered_Image_Image.axes.imshow(result_img, extent=[
                0.5, (result_img.shape[1]), 0.5, (result_img.shape[0]+0.5)])
            self.Canvas_for_Filtered_Image_Image.draw()
            cv2.imwrite(self.saved_img, result_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()
        else:
            filtered_img = cv2.medianBlur(original_img, self.fig_size)
            self.Canvas_for_Filtered_Image_Image.axes.imshow(filtered_img, cmap='gray', extent=[
                0.5, (filtered_img.shape[1]), 0.5, (filtered_img.shape[0]+0.5)])
            self.Canvas_for_Filtered_Image_Image.draw()
            cv2.imwrite(self.saved_img, filtered_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()

    def TwoD_Filter(self):
        #original_img = cv2.imread(file_path)
        original_img = self.im.copy()
        Kernal = np.ones((5, 5), np.float32) / 25
        if len(original_img.shape) == 3:
            converted_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            filtered_img = cv2.filter2D(converted_img, -1, Kernal)
            self.Canvas_for_Filtered_Image_Image.axes.imshow(filtered_img, extent=[
                0.5, (filtered_img.shape[1]), 0.5, (filtered_img.shape[0]+0.5)])
            self.Canvas_for_Filtered_Image_Image.draw()
            cv2.imwrite(self.saved_img, filtered_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()
        else:
            filtered_img = cv2.filter2D(original_img, -1, Kernal)
            self.Canvas_for_Filtered_Image_Image.axes.imshow(filtered_img, cmap='gray', extent=[
                0.5, (filtered_img.shape[1]), 0.5, (filtered_img.shape[0]+0.5)])
            self.Canvas_for_Filtered_Image_Image.draw()
            cv2.imwrite(self.saved_img, filtered_img)
            self.freq_domain(self.saved_img)
            self.draw_freq_filtered_Img()

    def fouier_transform(self, img):
        img_dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(img_dft)
        return dft_shift

    def Box_Filter(self):
        original_img = self.im.copy()
        if len(original_img.shape) == 3:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        filtered_spatial_image = cv2.boxFilter(
            original_img, -1, (10, 10), normalize=True)
        self.Canvas_for_Filtered_Image_Image.axes.imshow(filtered_spatial_image, cmap='gray', extent=[
            0.5, (filtered_spatial_image.shape[1]), 0.5, (filtered_spatial_image.shape[0]+0.5)])
        self.Canvas_for_Filtered_Image_Image.draw()

        dft_shift = self.fouier_transform(filtered_spatial_image)
        filtered_frequency_image = np.log(np.abs(dft_shift)+1)

        self.Canvas_for_Frequency_Domain_filteredImage.axes.imshow(filtered_frequency_image, cmap='gray', extent=[
            0.5, (filtered_frequency_image.shape[1]), 0.5, (filtered_frequency_image.shape[0]+0.5)])
        self.Canvas_for_Frequency_Domain_filteredImage.draw()

    def Laplacian_Filter(self):
        original_img = self.im.copy()
        if len(original_img.shape) == 3:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        filtered_spatial_image = cv2.Laplacian(original_img, -1, (10, 10))
        self.Canvas_for_Filtered_Image_Image.axes.imshow(filtered_spatial_image, cmap='gray', extent=[
            0.5, (filtered_spatial_image.shape[1]), 0.5, (filtered_spatial_image.shape[0]+0.5)])
        self.Canvas_for_Filtered_Image_Image.draw()

        dft_shift = self.fouier_transform(filtered_spatial_image)
        filtered_frequency_image = np.log(np.abs(dft_shift)+1)

        self.Canvas_for_Frequency_Domain_filteredImage.axes.imshow(filtered_frequency_image, cmap='gray', extent=[
            0.5, (filtered_frequency_image.shape[1]), 0.5, (filtered_frequency_image.shape[0]+0.5)])
        self.Canvas_for_Frequency_Domain_filteredImage.draw()

    def Frequency_Filter(self):
        #original_img = cv2.imread(file_path)
        original_img = self.im.copy()
        if len(original_img.shape) == 3:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            #original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        img_dft = np.fft.fft2(original_img)
        dft_shift = np.fft.fftshift(img_dft)

        if self.index == 2:
            filtered_dft_shift = self.highPassFiltering(dft_shift, 500)
        elif self.index == 1:
            filtered_dft_shift = self.lowPassFiltering(dft_shift, 500)

        filtered_frequency_image = np.log(np.abs(filtered_dft_shift)+1)

        self.Canvas_for_Frequency_Domain_filteredImage.axes.imshow(filtered_frequency_image, cmap='gray', extent=[
            0.5, (filtered_frequency_image.shape[1]), 0.5, (filtered_frequency_image.shape[0]+0.5)])
        self.Canvas_for_Frequency_Domain_filteredImage.draw()

        # Move the frequency domain from the middle to the upper left corner
        idft_shift = np.fft.ifftshift(filtered_dft_shift)
        filtered_spatial_image = np.fft.ifft2(
            idft_shift)  # Fourier library function call
        filtered_spatial_image = np.abs(filtered_spatial_image)
        self.Canvas_for_Filtered_Image_Image.axes.imshow(filtered_spatial_image, cmap='gray', extent=[
            0.5, (filtered_spatial_image.shape[1]), 0.5, (filtered_spatial_image.shape[0]+0.5)])
        self.Canvas_for_Filtered_Image_Image.draw()

    # Transfer parameters are Fourier transform spectrogram and filter size
    def highPassFiltering(self, img, size):
        h, w = img.shape[0:2]  # Getting image properties
        # Find the center point of the Fourier spectrum
        h_center, w_center = int(h/2), int(w/2)
        # Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
        img[h_center-int(size/2):h_center+int(size/2),
            w_center-int(size/2):w_center+int(size/2)] = 0
        return img

    # Transfer parameters are Fourier transform spectrogram and filter size
    def lowPassFiltering(self, img, size):
        h, w = img.shape[0:2]  # Getting image properties
        # Find the center point of the Fourier spectrum
        h_center, w_center = int(h/2), int(w/2)
        # Define a blank black image with the same size as the Fourier Transform Transfer
        img2 = np.zeros((h, w), np.uint8)
        # Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
        img2[h_center-int(size/2):h_center+int(size/2),
             w_center-int(size/2):w_center+int(size/2)] = 1
        # A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.
        img3 = img2*img
        return img3

    def draw_freq_filtered_Img(self):
        self.Canvas_for_Frequency_Domain_filteredImage.axes.imshow(self.magnitude_spectrum, cmap='gray', extent=[
            0.5, (self.magnitude_spectrum.shape[1]), 0.5, (self.magnitude_spectrum.shape[0]+0.5)])
        self.Canvas_for_Frequency_Domain_filteredImage.draw()

    def read_input_image(self):
        img = cv2.imread(self.fileName, 0)
        return img

    def flaten_image(self):
        img = self.read_input_image()
        img = np.asarray(img)
        flat_img = img.flatten()  # flats the matrix of img into 1D array
        #self.original_histogram_canvas.plt.hist(flat_img, bins=50) ####
        # print(flat_img)
        return flat_img

    def cummulative_sum(self, a):
        a = iter(a)
        b = [next(a)]

        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def original_histogram(self, bins):
        img = self.flaten_image()
        histogram = np.zeros(bins)
        for pixel in img:
            histogram[pixel] += 1
        # print(histogram)
        return histogram

    def plot_original_histogram(self):
        hist = self.original_histogram(256)
        print(hist.shape)
        self.ui.Frequency_Domain_Title.setText(self.translate(
            "MainWindow", "<html><head/><body><p align=\"center\">Original Histogram</p></body></html>"))
        self.Canvas_for_Frequency_Domain_Image.setVisible(False)
        self.Canvas_for_Original_Histogram.setVisible(True)
        self.Canvas_for_Original_Histogram.axes.plot(hist)

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
        self.Canvas_for_Frequency_Domain_filteredImage.setVisible(False)
        self.Canvas_for_Equalized_Histogram.setVisible(True)
        self.Canvas_for_Equalized_Histogram.axes.plot(equalized_hist)

    def equalized_img(self):
        equalized_hist = self.equalized_histogram()
        equalized_img = equalized_hist[self.flaten_image()]
        original_img = self.read_input_image()
        equalized_img = np.reshape(equalized_img, original_img.shape)
        self.Canvas_for_Filtered_Image_Image.setVisible(False)
        self.Canvas_for_Equalized_Image.setVisible(True)
        self.Canvas_for_Equalized_Image.axes.imshow(equalized_img, cmap='gray', extent=[
            0.5, (equalized_img.shape[1]), 0.5, (equalized_img.shape[0]+0.5)])

    def show_histogram(self):
        # hide the filters layout
        # show the histogram layout
        visibility = self.show_or_hide_widgets()
        if visibility:
            return
        self.Canvas_for_Frequency_Domain_Image.setVisible(False)
        self.Canvas_for_Filtered_Image_Image.setVisible(False)
        self.Canvas_for_Frequency_Domain_filteredImage.setVisible(False)
        self.Canvas_for_Equalized_Image.axes.cla()
        self.Canvas_for_Equalized_Histogram.axes.cla()
        self.Canvas_for_Original_Histogram.axes.cla()
        self.ui.Filtered_Image_Title.setVisible(True)
        self.ui.Frequency_Domain_filtered_Title.setVisible(True)
        self.ui.Filtered_Image_Title.setText(self.translate(
            "MainWindow", "<html><head/><body><p align=\"center\">Equalized Image</p></body></html>"))
        self.ui.Frequency_Domain_filtered_Title.setText(self.translate(
            "MainWindow", "<html><head/><body><p align=\"center\">Equalized Histogram</p></body></html>"))
        self.ui.Frequency_Domain_Title.setText(self.translate(
            "MainWindow", "<html><head/><body><p align=\"center\">Original Histogram</p></body></html>"))
        self.widgets_flag = 1
        self.equalized_img()
        self.plot_original_histogram()
        self.plot_equalized_histogram()
        self.Canvas_for_Original_Histogram.setVisible(True)
        self.Canvas_for_Equalized_Image.setVisible(True)
        self.Canvas_for_Equalized_Histogram.setVisible(True)
        self.ui.splitter_2.setSizes([100, 20])

    def show_filters(self):
        visibility = self.show_or_hide_widgets()
        if visibility == 0:
            return
        self.Canvas_for_Original_Histogram.setVisible(False)
        self.Canvas_for_Equalized_Image.setVisible(False)
        self.Canvas_for_Equalized_Histogram.setVisible(False)
        self.Canvas_for_Filtered_Image_Image.axes.cla()
        self.Canvas_for_Frequency_Domain_filteredImage.axes.cla()
        self.ui.Filtered_Image_Title.setVisible(True)
        self.ui.Frequency_Domain_filtered_Title.setVisible(True)
        self.ui.Frequency_Domain_Title.setText(self.translate(
            "MainWindow", "<html><head/><body><p align=\"center\">Frequency Domain</p></body></html>"))
        self.ui.Filtered_Image_Title.setText(self.translate(
            "MainWindow", "<html><head/><body><p align=\"center\">Filtered Image</p></body></html>"))
        self.ui.Frequency_Domain_filtered_Title.setText(self.translate(
            "MainWindow", "<html><head/><body><p align=\"center\">Frequency Domain</p></body></html>"))
        self.choose_filter()
        self.Canvas_for_Frequency_Domain_Image.setVisible(True)
        self.Canvas_for_Filtered_Image_Image.setVisible(True)
        self.Canvas_for_Frequency_Domain_filteredImage.setVisible(True)
        self.ui.splitter_2.setSizes([100, 20])

    def show_or_hide_widgets(self):

        if self.Canvas_for_Frequency_Domain_filteredImage.isVisible():
            self.ui.Filtered_Image_Title.setVisible(False)
            self.ui.Frequency_Domain_filtered_Title.setVisible(False)
            self.Canvas_for_Filtered_Image_Image.setVisible(False)
            self.Canvas_for_Frequency_Domain_filteredImage.setVisible(False)
            self.ui.splitter_2.setSizes([400, 20])
            return 0

        elif self.Canvas_for_Equalized_Histogram.isVisible():
            self.ui.Filtered_Image_Title.setVisible(False)
            self.ui.Frequency_Domain_filtered_Title.setVisible(False)
            self.Canvas_for_Equalized_Image.setVisible(False)
            self.Canvas_for_Equalized_Histogram.setVisible(False)
            self.ui.splitter_2.setSizes([400, 20])
            return 1


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ApplicationWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyside2())
    MainWindow.show()
    sys.exit(app.exec_())
