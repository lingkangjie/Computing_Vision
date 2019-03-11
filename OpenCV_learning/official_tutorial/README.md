# Knowledge Elements for Learning OpenCV-3.5
There are two main purposes for us need to extract knowledge elements when we try to learn something new.
Firstly and essentially, we always forget what we have learnt. Secondly, as I need to set up new OpenCV frequently.
If there exists a reference for me to check the details and structrues of a project, it will become more convenient.


For each project, I will write knowledeg elements follow the rules:
1. hierarchial knowledge object from abstraction to concrete
2. [abstraction]:[detials]


Notes: All there projects come from OpenCV team, reference: https://github.com/opencv/opencv/tree/master/samples/cpp

## build OpenCV from source
1. download OpenCV from official website
2. download Cmake
3. Cmake configure key points: 
step 1: click "where is the source code"
step 2: click "where to build the binaries( I mkdir a fold named "build" on opencv main folder)"
step 3: click "Configure", where "64" mean will generate  x64 bits .sln project
After generate, I choose NOT to build the test, and more configuration please reference the official website.
As Cmake will generate .sln project, then use VS to build the project. That is all.
After build, choose the "INSTALL" project and click build it, it will automatically install opencv in python if you
at first have python with Matplotlib and Numpy lib in your computer.

## application_trace.cpp
Total flow: read a video, and convert each frame to gray level, and apply canny operation to it.

1. input interface for command line: CommnadLineParser: argument, positional argument, default value, help message
2. read video file: VideoCapture: open
3. get the video file properties: nframes,width,height
4. OpenCV Trace macro for console output: CV_TRACE_REGION,CV_TRACE_REGION_NET
5. color space transformation:cvtColor
6. image edge detetor: Canny
7. a loop to process video frame by frame
5. a ESC key to quit: waitKey(1) == 27

## bgfg_segm.cpp
Goal: use KNN and MOG(mixtrue of gaussian model) to do background segmentation for frame in video.
1. input interface for command line: CommnadLineParser: using __--file___name="D:\xxx\xxxx.avi",NOT --file_name "D:\xxx\xxx.avi"
2. if the static video file is not provided, open camera: VideoCaputer: open
3. feed data to a model: model->apply
4. image show, data copy: copyTo
5. background segmentation algorithms: createBackgroundSubtractorKNN(), and createBackgroundSubtractorMOG2()
5. defind some keys to control the interactive process:'q','s',''
6. check camera ID: use OpenCVDeviceEnumerator project
