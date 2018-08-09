import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

VISUALIZE = False


class LaneFinder(object):
    def __init__(self, params):
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = params.get('nwindows', 9)
        # Set the width of the windows +/- margin
        self.margin = params.get('margin', 100)
        # Set minimum number of pixels found to recenter window
        self.minpix = params.get('minpix', 50)
        # Set max number of pixels for rediscovering the lane pixels with sliding window search
        self.rediscover_thr = params.get('rediscover_thr', 50)

        # Polynomial fit values from the previous frame
        self.left_fit = None
        self.right_fit = None
        # self.left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
        # self.right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])
        return

    def process(self, binary_warped):
        if self.left_fit is None or self.right_fit is None:
            # Search with sliding window at the fist time
            leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)
        else:
            # Search in a margin around the previous line position,
            leftx, lefty, rightx, righty = self.search_around_poly(binary_warped)
            if len(leftx) < self.rediscover_thr or len(rightx) < self.rediscover_thr:
                # Search again with sliding window when losing track of lines
                leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)

        self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        return

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        if VISUALIZE:
            # Create an output image to draw on and visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        margin = self.margin
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if VISUALIZE:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        self.nonzeroy = nonzeroy
        self.nonzerox = nonzerox
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        return leftx, lefty, rightx, righty

    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = self.margin
        left_fit = self.left_fit
        right_fit = self.right_fit

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        self.left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                              left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                              left_fit[1]*nonzeroy + left_fit[2] + margin)))
        self.right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                               right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                               right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds]
        rightx = nonzerox[self.right_lane_inds]
        righty = nonzeroy[self.right_lane_inds]
        self.nonzeroy = nonzeroy
        self.nonzerox = nonzerox
        return leftx, lefty, rightx, righty

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        # Fit a second order polynomial to each with np.polyfit() ###
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        return

    def get_poly_plot(self, img_shape):
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        # Calc both polynomials using ploty, left_fit and right_fit ###
        left_fit = self.left_fit
        right_fit = self.right_fit
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        return left_fitx, right_fitx, ploty

    def visualize(self, binary_warped):
        left_fitx, right_fitx, ploty = self.get_poly_plot(binary_warped.shape)
        margin = self.margin

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])

        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        # Plot the polynomial lines onto the image
        def gen_points(x):
            pts = np.int_(np.dstack((x, ploty)))
            pts = pts.reshape((-1, 1, 2))
            return [pts]
        cv2.polylines(out_img, gen_points(left_fitx), False, (255, 255, 255), thickness=1)
        cv2.polylines(out_img, gen_points(right_fitx), False, (255, 255, 255), thickness=1)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
