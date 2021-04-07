### Video Data

The videos used as most of the data in this folder were created by the author with the [.upCam Cyclone HD eco](https://www.upcam.de/upcam-cyclone-hd-eco-ip-kamera-innen) camera model. Video data was collected in daily batches, 640x352 resolution and 2.5-5.0 FPS (varies due to camera) from the home office. Preprocessing was kept to a minimum; merely the resolution was downgraded and the frames were set to a constant amount (5) for each second, causing frame duplications. This was done with ffmpeg. 

The dataset (*/upCam/*) is organized as followed: The processed dataset has has its videos organized by day (the date was not relevant at the time so they are simply ordered in ascending order by their original date) and each day-directory contains the 24 video files (one for each hour). **Note:** Day `00-04` was captured between the 26th of September and the 30th of September (2020), while `05-09` was captured on the 15th to the 19th of November.

Code relevant to these preprocessing steps can be found in the source code repository in `src/io/preprocessing`. Due to space reasons, the raw video files have been removed from the repository.

#### Datasets
- **eval:** Directory for all (labeled) testing datasets created by the upCam.
    - *labels:* True labels (boolean) of each sample/frame
		- `.csv` labels per frame
		- `.md` labels per event/frame ranges
    - *preprocessed_128_64:* 5 FPS, 128x64 resolution
- **upCam:** Directory for all (unlabeled) training datasets created by the upCam.
    - *preprocessed_128_64:* 5 FPS, 128x64 resolution
