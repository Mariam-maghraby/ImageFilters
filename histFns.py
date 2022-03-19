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
