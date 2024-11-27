from moviepy import ImageSequenceClip
import os

def create_video_from_images(image_folder, output_file, fps=24):
    """Create an MP4 video from a sequence of PNG images in a folder."""
    # Get all PNG files in the folder
    image_files = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png")])
    
    # Create the video clip
    clip = ImageSequenceClip(image_files, fps=fps)
    
    # Write the video file
    clip.write_videofile(output_file, codec='libx264')

# Example usage
create_video_from_images("test_output", "output_video.mp4")
