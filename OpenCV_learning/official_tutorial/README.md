# Knowledge Elements for Learning OpenCV-3.5
There are two main purposes for us need to extract knowledge elements when we try to learn something new.
Firstly and essentially, we always forget what we have learnt. Secondly, as I need to set up new OpenCV frequently.
If there exists a reference for me to check the details and structrues of a project, it will become more convenient.


For each project, I will write knowledeg elements follow the rules:
1. Hierarchial knowledge object from abstraction to concrete
2. Abstraction:detials


Notes: All there projects come from OpenCV team, reference: https://github.com/opencv/opencv/tree/master/samples/cpp

## build OpenCV from source
1. Download OpenCV from official website
2. Download Cmake
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

1. Input interface for command line: CommnadLineParser: argument, positional argument, default value, help message
2. Read video file: VideoCapture: open
3. Get the video file properties: nframes,width,height
4. OpenCV Trace macro for console output: CV_TRACE_REGION,CV_TRACE_REGION_NET
5. Color space transformation:cvtColor
6. Image edge detetor: Canny
7. A loop to process video frame by frame
8. A ESC key to quit: waitKey(1) == 27

## bgfg_segm.cpp
Goal: use KNN and MOG(mixtrue of gaussian model) to do background segmentation for frame in video.
1. Input interface for command line: CommnadLineParser: using __--file___name="D:\xxx\xxxx.avi",NOT --file_name "D:\xxx\xxx.avi"
2. If the static video file is not provided, open camera: VideoCaputer: open
3. Feed data to a model: model->apply
4. Image show, data copy: copyTo
5. Background segmentation algorithms: createBackgroundSubtractorKNN(), and createBackgroundSubtractorMOG2()
5. Defined some keys to control the interactive process:'q','s',''
6. Check camera ID: use OpenCVDeviceEnumerator project

## camshiftdemo.cpp
Goal: object tracking using the mean-shift tracking algorithm
1. A help function: parse.has("help"):call help()
2. Open camera, and check whether success
3. Generate two windows:highgui.hpp:nameWindow
4. Add track bar to window: createTrackbar: track bar will return values stored in &vmin, &vmax,&smin
5. A mouse callback function to draw box on video: setMouseCallback: callback function,onMouse()
6. InRange():Checks if array elements lie between the elements of two other arrays
7. More ....(to do)

## cloning_demo.cpp
Goal: demonstrates how to use OpenCv seamless cloning
1. folder_path: in windows,use std::string floder = "D:\\tmp\\xxx\\"
2. Six types of seamless cloning:seamlessClone(1)(Normal Cloning),seamlessClone(2)(Mixed Cloning), seamlessClone(3)(Monochrome Transfer),
colorChange(),illuminationChange(),textureFlattening()
3. Three images to generate one image: source.png, destination.png,mask.png
4. Check files whether exist:samples::findFile
5. Read image with color: imread(IMREAD_COLOR)
6. Check whether image is empty
7. imwrite

## connected__components.cpp
Goal: find connected components in an image and display them with different color
1. argv[0]:parser.get<string>(0):in .sln project,debug parameter: D:\XXX\official_tutorial\data\color_change\reference.png
2. Add a trackbar to control the threshold level: how trackbar calls the callback function? At first, just display the processed image
in image window with default threshold. Then the program will in loop, as trackbar event happens, the callback function will be called
just at once.
3. on_trackbar function:give a label for a pixel, if pixels have the same label will be denoted by same color.
4. Vec3b data structure.Get pixel fomr an image: image.at

## contours2.cpp
Goal: Draw an image, extract contours in it, and show 
1. Draw in image: line(),ellipse()
2. Coordinates in vector of vector data structure:vector<vector<Point> >
3. Find contours: findContours()
4. appromimates a ploygonal curve(s):approxPolyDP()
5. Draw contours using contours points: drawContours()

## convexhull.cpp
Goal: show how to use convexHull() function
1. convexHull(): find the convex hull of a 2D point set using the Sklansky's algorithm
2. Generate random number: rng.uniform()
3. Use the random as center point for circle and draw a circle for visualizing: circle()
4. Concatenate convex hull points using line(): line()
5. Wait a loop and exit: 27==key,'q','Q'

## cout_mat.cpp
Goal: show how to serial out capabilities of cv::Mat, show Mat basic operations
1. Eye matrix: Mat::eye()
2. Directly cout Mat: cout<< I
3. Generate a Mat and fill it with random number: randu()
4. Convert cpp style of Mat to other forms: FMT_MATLAB, FMT_PYTHON, FMT_NUMPY, FMT_CSV, FMT_C
5. Generate a 2D point, a 3D point and cout them: typename Point2f, typename Point3f
6. Convert vector or Mat 
7. genetate a vector of Point2f(typename)

## create_mask.cpp
Goal: use mouse to mask an image by hand. Left mouse button, set a point to create mask shape.
Right mouse button, create mask from points. Middle mouse button, reset.
1. CommandLineParser, check whether exist, read image.
2. Mouse event handling: setMouseCallback(): EVENT_LBUTTONDOWN, EVENT__LBUTTONUP, EVENT__RBUTTONDOWN,
EVENT_RBUTTONUP, EVENT__MBUTTONDOWN
3. A polylines() function to recolor the line() color
4. Use fillPoly() to fill mask Mat with 1, and mask(bitwise_and() function) source image

## dbt_face_detection.cpp
### Backgound Knowledge of LBP
LBP is a type of visual descriptor used for classification in computer vision. The basic idea of LBP is, as "local" meaning
indicates, LBP will be applied to cells(e.g. 16X16 pixels for each cell) of image. For each pixel in a cell, LBP compares the
pixel to each of its 8 neighors(on its(denoted as C) left-top,left-middle,etc.,total 8 directions,denoted as N0, N1,..., N7).
If N0>C, then N0=1. Otherwise, N0=0. N1,...,N7 will be operated as the same way. So for a pixel, will has a binary number
(N0N1N2..N&,e.g. 11100011). This 8-bits binary number usually converted to decimal for convenience. So, a cell(16x16 pixels) has
16x16 decimal number. At last, computer the histogram over the cell, normalized this histogram, and this histogram can be 
seen as a 256-dimensional feature vector. So a cell will has a feature vector coming from histogram computing, concatenated all
histograms of all cells as final features for the image.  
### LBP Cascade Classifier
1. **LBP Labelling**: A label as a string of binary numbers is assigned to each pixel of each image.
2. **Feature Vector**: Image is divided into sub-regions and for each
sub-region, histogram of labels is constructed. Then a feature vector is
formed by concatenating the sub-region histograms into a large histogram.
3. **AdaBoost Learning**: Strong classifier is constructed using gentle
AdaBoost to remove redundant information from feature vector.
4. **Cascade of Classifier**: The cascades of classifier are formed from the
features obtained by the gentle AdaBoost algorithm. Sub-regions of the image
is evaluated starting from simpler classifier to strong classifier. If on any
stage classifier fails, that region will be discarded from further iterations.
Only the facial region will pass all the stages of the classifier.  
### the Pros and Cons of LBP
Pros: Computationally simply and fast, robust to local illumination changes
and occlusion.  
Cons: Less accurate, and high false positive rate.  

### source code analysis
Goal: use LBP(local binary pattern) features and cascade classifier for face detection
1. Read pre-train LBP feature for human face, in official OpenCV project, are
located in /data/lbpcascades/lobcascade_frontalface.xml
2. Make a Adapter class for DetectionBasedTracker::IDetector and
CascadeClassifier.
3. More...(to do)

## delaunay2.cpp
Goal: demonstrates itetative construction delaunny triangulation and voronoi
tessellation.
**Background**: Given a set of points in a 2D plane, we want many triangles
with the points as vertices to divide the plane into many sub-plane. This
process called triangulation, and delaunny triangulation is a kind of such
triangulation. A set of points can have many possible triangulations, but
Delaunay triangulation stands out because it has some nice properties. In a
Delaunay triangulation, triangles are chosen such that no point is inside the
circumcircle of any triangle. What is a Voronoi diagram? Given a set of
points, a Voronoi diagram partitions the space such that the boundary lines
are equidistant from neighboring points. That is to say that every boundary
line passes through the center of two points. If you connect the points in
neighboring Voronoi regions, you get a Delaunay triangulation.  
1. Subdiv2D: A clss used to perform various planar subdivision on a set of 2D
points(represented as vector of Point2f)
2. Subdiv2D.locate: Return the location of a point with a Delaunay
triangulation.
3. draw_subdiv_point(): print a point(two number) as a circle to visualize.
4. draw_subdiv(): Given a Subdiv2D object, extract all triangles in
it(triangleList),for each point in a triangle, draw line between them.
5. paint_voronoi():Given a Subdiv2D object, extract all Voronoi facets( output
form is std::vector<std::vector<Point2f> >) and its center(Voronoi facets
center points). For each small facets, use polylines() to fill the facet area.
For each Voronoi center point, use circle() to visualize. 

## demhist.cpp
Goal: demonstrates the use of calcHist() -- histogram creation, and how to
change the image brightness and contrast.
1. A help function, read image, show image.
2. A Trackbar to control values from user, and these values are global values
called _brightness and _contrast respectively.
3. A callback function: updateBrightnessConstrast() to deal with event.
4. Brightness and contrast: new_pixel_value = a(brightness) times old_value
plus b(contrast).
5. updateBrightnessConstrast(): Mat::convertTo(): convert source values to the target data type with a scale,
here is used to map original image to new image with new brightness and
contrast.
6. updateBrightnessConstrast(): calcHist(),normalize(),cvRound() -- Rounds
floating-point number to nearest integer
7. updateBrightnessConstrast(): use rectangle() to draw histogram bins to an
new image Mat called histImage. Very fundamental and elaborately, as
calcHist() function just outputs *hist* variable, not visualize. It is a good
example for us to learn plot histogram.

## detect_blob.cpp
Goal: SimpleBlobDetector to detect blob in image  
What is blob? A Blob is a group of connected pixels in an image that share some common property ( E.g grayscale value ). The goal of blob detection is to identify and mark these regions.  
1. An CommandLineParser, check files, read image, determinate whether image is
empty.
2. SimpleBlobDetector: set default parameters, generate a vector(pBlob) for
containing params blob and generate an iterator for this array.
3. Generate a total of 65536 vectors containing Vec3b object(vector< Vec3b>) for color
palette, and the color is initialized by random.
4.How to change the parameter blob? Firstly, push_back the default parameter
blob(pDefaultsBLOB) to pBlob(a vector),then use vector::back() to get it and set another parameter
elements. 
5. Generate an iterator for descriptor array for BLOB(typeDesc, just a vector
containing 6 "BLOB" strings),and iterates it for handling each parameter
setting in loop.
6. A Legende() function outputs a label to describe each params setting blob:
how to cast num to str in SimpleBlobDetector? String s = static_cast< const
ostringstream&>(ostringstream() << inputParamBlob.param_element).str(). Note
that Legende() called by reference.
7. SimpleBlobDetector::create(): input is params, called by reference: return
a pointer to params
8. drawKeypints(): Draw keypoints in imge.
9. sbd->detect(img,KeyImg,Mat()): KeyImage, a vector contains KeyPoint( has
point coordinate and circle size) 

## dft.cpp
Goal: demonstrates the use of Discrete Fourier Transform(DFT)
1. Basic using: getOptimalDFTSize(), copyMakeBorder(),
merge(),dft(),split(),magnitude(),log(),normalize()

## distrans.cpp
Goal: demonstrates the use of distance transform function between edge images
1. A global variable(distType0) to save keyboard input. Keyboard
input:'c','1','2','5','5','0','v','p',''(space key).And the Space control a
loop through all the modes.
2. In Trackbar callback function(onTrackbar()), deals with the distType0
variable.
3. cv::distanceTransform calculates the approximate or precise distance from
every binary image pixel to the nearest zero pixel. The ouput is a 8-bit or
32-bit floating-point single-channel image of the same size as src.

## drawing.cpp
Goal: demonstrates OpenCV drawing and text output functions
1. How to generate random color(customized randomColor()), random point location.
1. line():line(img,Point pt1, Point pt2, color,int thickness,int line_type,int
shfit)
2. arrowedLine():draws a arrow segment pointing fromthe first point to the
second one.
3. rectangle(), drawMarker(): loop 100 times.
4. Loop ellipse() 100 times.
5. Loop polylines() 100 times: draws polygonal curves.
6. Loop fillPloy() 100 times: fills the area bounded by one or more polygons.
7. Loop circel() 100 times.
8. Loop putText() 100 times: put the text in image.
9. Fade-out features: make two Mat, image2 = image - Scalar::all(i),then
display and update image2.

## edge.cpp
Goal: edge detection using Canny detector with Sobel gradient and Canny detector with custom
gradient(Scharr)
1. Two trackbar and one callback function(onTrackbar)
2. In callback function: preform Canny operation, show results

## em.cpp
Goal: use Expectation Maximization(EM) algorithm to classification. From OpenCV Machine Learning
Library. I found that Lib is more isolated from OpenCV core lib, That is a
good lib for us to implement ML algorithms from scratch.
1. As in OpenCV, the basic data structure is Mat. For ML algorithms,matrices
are more common. So we need a new aspect view to watch Mat. In OpenCV ML, from
the aspect of *rowRange()* segmentation,I
found OpenCV team use "channel" to represent "column" in general ML
concepts(like in MATLAB). So em.cpp first generate a *samples* Mat, its row
represents samples, col represents dim. Then reshape *samples* using
*samples.reshape(2,0)*: reshape(cn,rows),cn(new number of channels, if is
0,the number of channels remains the same), rows(new number of rows,if is 0,
the number of rows remains the same). After reshape, *samples* becomes
(row2=100,cols=1). After segmentation done, *reshape(1,0)*, *samples*
becomes(rows=100,cols=2).
2. Then split samples into 4 parts(from row aspect), represents 4 class input
data(*sample_part*). For each class, fill *sample_part* with random(*randn*).
3. Reshape *samples* to (rows=100,cols=2) 
4. When train done, use the EM model to classify every image pixel for
visualizing.
5. Draw 100 samples(2 dims) location sample by sample.

## example.cpp
Goal: draw text on video
1. Open video input stream: VideoCaputer::open(0): video_stream >> *image*: Then
draw something on *image*, and show *image*.

## facedetect.cpp
Goal: cascade face detection based on Haar or LBP features.  
**cascade**: the primary trained classifier such as frontal face.  
**nested-cascade**: an optional secondary classifier such eyes.
1. Make two CascadeClassifier called *cascade* and *nestedCascade*,and load
cascade from file using *CascadeClassifier::load*.
2. Check whether success for open camera, loading files, etc.
3. detectAndDraw():input the image frame, two CascadeClassifier, image scale,
whether to flip. Convert image to gray, resize it(as we use *scale* variable),
equalizeHist() it.
4. detectAndDraw(): time tick: *(double)getTickCount()*
5. detectAndDraw(): if flip opened, the *face2* from flipped image will be
push bach to *faces*(not flip).
6. detectAndDraw(): If detect faces(vector in vector,(x,y,width,height)),draw
circle or rectangle to denote where is the face location. And use this area as
the input to *nestedCascade.detectMultiScale()*, at last ,draw the results(eye(s)) from *nestedCascade*.

## facial_features.cpp
Goal:


