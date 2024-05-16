import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
import os


# Class for storing lane line predictions
class LanePredictor():
    def __init__(self):
        self.recent_predictions = []
        self.average_prediction = []
        self.has_saved_mask = False 


def detect_lane_lines(input_image, output_mask_dir='OutputMask'):
    # Prepare image for input into neural network
    resized_image = imresize(input_image, (80, 160, 3))
    resized_image = np.array(resized_image)
    neural_input = resized_image[None,:,:,:]

    # Predict lane lines with neural network and scale up values
    neural_prediction = lane_line_model.predict(neural_input)[0] * 255

    # Append new prediction to the list for averaging
    lane_predictor.recent_predictions.append(neural_prediction)
    # Use only the last five predictions for averaging
    if len(lane_predictor.recent_predictions) > 5:
        lane_predictor.recent_predictions = lane_predictor.recent_predictions[1:]

    # Compute the average of recent predictions
    lane_predictor.average_prediction = np.mean(np.array(lane_predictor.recent_predictions), axis = 0)

    # Create blank channels for red and blue, merge with the green channel
    empty_channels = np.zeros_like(lane_predictor.average_prediction).astype(np.uint8)
    green_lanes_image = np.dstack((lane_predictor.average_prediction/2, lane_predictor.average_prediction/2, empty_channels))
   
    # Resize the lanes image to match the original input image dimensions
    full_size_lane_image = imresize(green_lanes_image, (video_clip.size[1], video_clip.size[0], 3))

    # Combine the lanes image with the original image
    combined_image = cv2.addWeighted(input_image, 1, full_size_lane_image, 1, 0)
    
    # Save the mask image, if not saved already
    if not lane_predictor.has_saved_mask:
        video_basename = os.path.basename(input_video_path)
        mask_file_path = os.path.join(output_mask_dir, video_basename.rsplit('.', 1)[0] + "_mask.png")
        mask_image = lane_predictor.average_prediction.astype(np.uint8)
        mask_image_resized = cv2.resize(mask_image, (video_clip.size[0], video_clip.size[1]))
        cv2.imwrite(mask_file_path, mask_image_resized)
        lane_predictor.has_saved_mask = True

    return combined_image


if __name__ == '__main__':
    input_video_path = input("Please enter the relative path of the video: ")
    model_file_path = input("Please enter the relative path of the model: ") 
    output_mask_dir = 'OutputMask'

    # Load the Keras model for predicting lane lines
    lane_line_model = load_model(model_file_path)

    # Initialize the LanePredictor object
    lane_predictor = LanePredictor()

    # Modify the 'detect_lane_lines' function to save the mask image
    def detect_lane_lines_with_mask(image):
        return detect_lane_lines(image, output_mask_dir=output_mask_dir)
    
    # Set the path for the output video file
    output_video_path = os.path.join('OutputVideo', os.path.basename(input_video_path))

    # Load the input video clip
    video_clip = VideoFileClip(input_video_path)

    # Apply the function to each frame of the video
    processed_video_clip = video_clip.fl_image(detect_lane_lines_with_mask)
    # Write the processed video clip to the output file
    processed_video_clip.write_videofile(output_video_path, audio=False)
    
