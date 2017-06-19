# %% Load
import os
import glob

basedir = 'vehicles/'
image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
    cars.extend(glob.glob(basedir+imtype+'/*'))
print('Number of cars found:', len(cars))

basedir = 'non-vehicles/'
image_types = os.listdir(basedir)
notcars = []
for imtype in image_types:
    notcars.extend(glob.glob(basedir+imtype+'/*'))
print('Number of not cars found:', len(notcars))

# %% Imports
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
%matplotlib inline

# %% Pipeline
# Function to return HOG features and visualization
# Similar objects like cars can have different colors and looking only on histogram will not generalize well
# but if we look at those object gradients they will be more similar and easier for classfier to generalize
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    result = hog(img, orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   transform_sqrt=True,
                   visualise=vis, feature_vector=feature_vec)

    if vis:
        features, hog_image = result
        return features, hog_image
    else:
        features = result
        return features, None


# Perform spatial binning (downsampling) to represent object pixels in reduced size feature vector.
def bin_spatial(img, size=(32, 32)):
    img_res = cv2.resize(img, size)
    color1 = img_res[:,:,0].ravel()
    color2 = img_res[:,:,1].ravel()
    color3 = img_res[:,:,2].ravel()
    return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

# Define a function to extract features from a image
def extract_features(files, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                    spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in files:
        image = mpimg.imread(file)
        file_features = single_img_features(image,
                                            color_space,
                                            spatial_size,
                                            hist_bins,
                                            orient,
                                            pix_per_cell,
                                            cell_per_block,
                                            hog_channel,
                                            spatial_feat,
                                            hist_feat,
                                            hog_feat,
                                            vis = False)

        features.append(file_features)
    return features

def convert_RGB(image_rgb, color_space):
    if color_space != 'RGB':
        if color_space == 'HSV':
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    else: image = np.copy(image_rgb)
    return image

def single_img_features(image, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis = False):
    img_features = []
    feature_image = convert_RGB(image, color_space)

    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            hog_img = []
            for channel in range(feature_image.shape[2]):
                found_features, channel_img = get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis, feature_vec=True)
                hog_features.append(found_features)
                hog_img.append(channel_img)
            hog_features = np.ravel(hog_features)
        else:
            hog_features, hog_img = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis, feature_vec=True)

        img_features.append(hog_features)

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    img_features = np.concatenate(img_features)
    if vis:
        return img_features, hog_img
    else:
        return img_features

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            window_list.append(((startx, starty), (endx, endy)))

    return window_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def search_windows(img, windows, classifier, scaler,
                   color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = single_img_features(test_img,
                                        color_space,
                                        spatial_size,
                                        hist_bins,
                                        orient,
                                        pix_per_cell,
                                        cell_per_block,
                                        hog_channel,
                                        spatial_feat,
                                        hist_feat,
                                        hog_feat,
                                        vis = False)

        test_features = scaler.transform(np.array(features).reshape(1,-1))
        prediction = classifier.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows

def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims= len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])

# %% TEST SINGLE IMG FEATURES
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

color_space = 'RGB' # check HSV or YCrCb
orient = 9 # number of orientation angles
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # can be 0,1,2 or "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

car_features, car_hog_image = single_img_features(car_image,
                                        color_space,
                                        spatial_size,
                                        hist_bins,
                                        orient,
                                        pix_per_cell,
                                        cell_per_block,
                                        hog_channel,
                                        spatial_feat,
                                        hist_feat,
                                        hog_feat,
                                        vis = True)
notcar_features, notcar_hog_image = single_img_features(notcar_image,
                                        color_space,
                                        spatial_size,
                                        hist_bins,
                                        orient,
                                        pix_per_cell,
                                        cell_per_block,
                                        hog_channel,
                                        spatial_feat,
                                        hist_feat,
                                        hog_feat,
                                        vis = True)

images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
titles = ['car image', 'car  HOG image', 'notcar_image', 'notcar HOG image']
fig = plt.figure(figsize=(12,3), dpi=100)
visualize(fig, 1, 4, images, titles)

# %% TRAIN CLASSIFIER
color_space = 'YCrCb' # check HSV or YCrCb
orient = 9 # number of orientation angles, improvement up to 9 then not
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # can be 0,1,2 or "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

t=time.time()
# n_samples = 1000
# random_idxs = np.random.randint(0, len(cars), n_samples)
input_cars = cars # np.array(cars)[random_idxs]
input_notcars = notcars # np.array(notcars)[random_idxs]

car_features = extract_features(input_cars,
                                color_space,
                                spatial_size,
                                hist_bins,
                                orient,
                                pix_per_cell,
                                cell_per_block,
                                hog_channel,
                                spatial_feat,
                                hist_feat,
                                hog_feat)

notcar_features = extract_features(input_notcars,
                                    color_space,
                                    spatial_size,
                                    hist_bins,
                                    orient,
                                    pix_per_cell,
                                    cell_per_block,
                                    hog_channel,
                                    spatial_feat,
                                    hist_feat,
                                    hog_feat)

print(round(time.time()-t,2), 'Seconds to compute features...')

X = np.vstack((car_features,notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear Support Vector CLASSIFIER
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
print(round(time.time()-t, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# %% TEST SLIDING WINDOW
searchpath = 'test_images/*'
example_images = glob.glob(searchpath)
images = []
titles = []
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
overlap = 0.5
for img_src in example_images:
    t = time.time()
    image = mpimg.imread(img_src)
    draw_image = np.copy(image)
    # Training data are .png images scaled 0 to 1 by mpimg.
    # When reading .jpg images we need to scale them to 0-1 as well.
    image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                        xy_window=(128, 128), xy_overlap=(overlap, overlap))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    window_image = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    images.append(window_image)
    titles.append('')
    print(time.time() - t, 'seconds to search boxes in one image', len(windows), 'windows')

fig = plt.figure(figsize=(8,12), dpi = 100)
visualize(fig, 5 ,2, images, titles)

# %% IMPROVE PREDICTION TYPE WITH HOG COMPUTATION FOR A WHOLE PICTURE
from scipy.ndimage.measurements import label
from functools import reduce
from collections import deque

class CarDetector():
    def __init__(self, normalizer, classifier, conf, heat_queue_len = 5):
        self.normalizer = normalizer
        self.classifier = classifier
        self.conf = conf
        self.heat_queue = deque(maxlen=heat_queue_len)

    def detect(self, img, is_png = False):
        draw_img = np.copy(img)
        if is_png:
            img = img.astype(np.float32)/255 # jpg normalization

        heatmaps = []
        for scale in conf['scales']:
            new_heat= self.heatmap(img, scale)
            heatmaps.append(new_heat)
        combined = reduce(np.add, heatmaps)

        out_heat = self.smoothen_heatmap(combined)
        labeled_detections = label(out_heat)
        out_img = self.draw_labeled_boxes(np.copy(draw_img), labeled_detections)
        return (out_img, out_heat)

    def smoothen_heatmap(self, heatmap):
        self.heat_queue.append(heatmap)
        out_heat = reduce(np.add, list(self.heat_queue))
        out_heat = self.threshold(out_heat, self.conf['heat_threshold'])
        return out_heat

    def heatmap(self, img, scale):
        conf = self.conf
        road_area = img[conf['y_start']:conf['y_stop'],:,:]
        road_area = convert_RGB(road_area, conf['color_space'])

        if scale != 1:
            road_shape = road_area.shape
            road_area = cv2.resize(road_area, (np.int(road_shape[1]/scale), np.int(road_shape[0]/scale)))

        hog1, hog_img1 = get_hog_features(road_area[:,:,0], conf['orient'], conf['pix_per_cell'], conf['cell_per_block'], feature_vec=False)
        hog2, hog_img2 = get_hog_features(road_area[:,:,1], conf['orient'], conf['pix_per_cell'], conf['cell_per_block'], feature_vec=False)
        hog3, hog_img3 = get_hog_features(road_area[:,:,2], conf['orient'], conf['pix_per_cell'], conf['cell_per_block'], feature_vec=False)

        window_size = 64
        cell_size = conf['pix_per_cell']
        cells_per_window = (window_size // cell_size) - 1
        cells_in_col = (road_area.shape[0] // cell_size) - 1
        cells_in_row = (road_area.shape[1] // cell_size) - 1
        cells_stride = 2 # Window have 64 cells = 8x8, stride of 2 gives 6/8 overlap.

        horizontal_steps =  (cells_in_row - cells_per_window) // cells_stride
        vertical_steps =  (cells_in_col - cells_per_window) // cells_stride

        heatmap = np.zeros_like(img[:,:,0])
        for h in range(horizontal_steps):
            for v in range(vertical_steps):
                feature_vector = []

                x_cell = h * cells_stride
                y_cell = v * cells_stride
                if conf['hog_feat'] == True:
                    hog_feat1 = hog1[y_cell:y_cell+cells_per_window, x_cell:x_cell+cells_per_window].ravel()
                    hog_feat2 = hog2[y_cell:y_cell+cells_per_window, x_cell:x_cell+cells_per_window].ravel()
                    hog_feat3 = hog3[y_cell:y_cell+cells_per_window, x_cell:x_cell+cells_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    feature_vector.append(hog_features)

                x_px = x_cell * cell_size
                y_px = y_cell * cell_size
                subimg = cv2.resize(road_area[y_px:y_px+window_size, x_px:x_px+window_size], (64, 64))
                if conf['spatial_feat'] == True:
                    spatial_features = bin_spatial(subimg, conf['spatial_size'])
                    feature_vector.append(spatial_features)

                if conf['hist_feat'] == True:
                    hist_features = color_hist(subimg, conf['hist_bins'])
                    feature_vector.append(hist_features)

                feature_vector = np.hstack(feature_vector).reshape(1,-1)
                feature_vector = self.normalizer.transform(feature_vector)
                prediction = self.classifier.predict(feature_vector)

                if prediction == 1:
                    x_real = np.int(x_px*scale)
                    y_real = np.int(y_px*scale) + conf['y_start']
                    window_real_size = np.int(window_size*scale)
                    x_real_end = x_real + window_real_size
                    y_real_end = y_real + window_real_size
                    heatmap[y_real:y_real_end, x_real:x_real_end] += 1

        return heatmap

    def threshold(self, heatmap, threshold):
        heatmap[heatmap < threshold] = 0
        return heatmap

    def draw_labeled_boxes(self, img, labels):
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bounding_box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img, bounding_box[0], bounding_box[1], (0,0,255), 6)
        return img

# %% TEST HEATMAPS
conf = {
    'y_start': 400
    ,'y_stop': 656
    ,'color_space': 'YCrCb'
    ,'pix_per_cell': 8
    ,'cell_per_block': 2
    ,'orient': 9
    ,'hog_channel': 'ALL'
    ,'hist_bins': 32
    ,'spatial_size': (32, 32)
    ,'spatial_feat': True
    ,'hist_feat': True
    ,'hog_feat': True
    ,'scales': [1.0, 1.5]
    ,'heat_threshold': 2
}

searchpath = 'test_images/*'
example_images = glob.glob(searchpath)
out_images = []
out_titles = []

for img_src in example_images:
    img = mpimg.imread(img_src)

    detector = CarDetector(normalizer = X_scaler, classifier= svc, conf = conf)
    out_img, out_heat = detector.detect(img, is_png = True)

    out_images.append(out_img)
    out_titles.append(img_src[-9:])
    out_images.append(out_heat)
    out_titles.append('heatmap ' + str(img_src[-9:]))

fig = plt.figure(figsize=(12,24), dpi = 100)
visualize(fig, 6,2, out_images, out_titles)

# %% TEST VIDEO
from moviepy.editor import VideoFileClip
from IPython.display import HTML

conf = {
    'y_start': 400
    ,'y_stop': 656
    ,'color_space': 'YCrCb'
    ,'pix_per_cell': 8
    ,'cell_per_block': 2
    ,'orient': 9
    ,'hog_channel': 'ALL'
    ,'hist_bins': 32
    ,'spatial_size': (32, 32)
    ,'spatial_feat': True
    ,'hist_feat': True
    ,'hog_feat': True
    ,'scales': [1.0 , 1.25, 1.5]
    ,'heat_threshold': 10
}

detector = CarDetector(normalizer = X_scaler, classifier= svc, conf = conf, heat_queue_len = 10)

# test_output = 'test.mp4'
# clip = VideoFileClip('test_video.mp4')
test_output = 'project.mp4'
clip = VideoFileClip('project_video.mp4')
test_clip = clip.fl_image(lambda img: detector.detect(img, is_png = True)[0])
#%time
test_clip.write_videofile(test_output, audio = False)

# HTML("""
# <video width="960" heigh="540" controls>
#     <source src="{0}">
# </video>
# """.format(test_output))
