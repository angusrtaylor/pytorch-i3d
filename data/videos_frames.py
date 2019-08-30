# Extracts RGB frames from HMDB videos 
import cv2
import os
from pathlib import Path

videos = [f.resolve() for f in Path('/largedata/i3d/videos').glob('**/*.avi')]
print(videos[:10])

new_dirs = []
for v in videos:
    print(v)
    new_dir = os.path.join(str(v)[:-4],'i')  # RGB FRAME
    os.makedirs(new_dir, exist_ok=True)
    new_dirs.append(new_dir)
    vidcap = cv2.VideoCapture(str(v.resolve()))
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        cv2.imwrite(os.path.join(new_dir, "frame%04d.jpg" % count), image)     # save frame as JPEG file
        count += 1


# Create list of folders
with open('hmdb_videos.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % v for v in new_dirs)