#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing.h>

#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_loader/jpeg_loader.h>
#include <dlib/threads/threads_kernel_shared.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace cv;
using namespace dlib;
using namespace std;
//Read points from text file
std::vector<Point2f> readPoints(string pointsFileName){
	std::vector<Point2f> points;
	ifstream ifs (pointsFileName.c_str());
    float x, y;
	int count = 0;
    while(ifs >> x >> y)
    {
        points.push_back(Point2f(x,y));

    }

	return points;
}
/*
void faceLandmarkDetection(dlib::array2d<unsigned char>& img, shape_predictor sp, std::vector<Point2f>& landmark)
{
  frontal_face_detector detector = get_frontal_face_detector();

  std::vector<dlib::rectangle> dets = detector(img);
  DLOG("Number of faces detected: %lu \n", dets.size());

  full_object_detection shape = sp(img, dets[0]);

  for (int i = 0; i < shape.num_parts(); ++i) {
    float x=shape.part(i).x();
    float y=shape.part(i).y();
    landmark.push_back(Point2f(x,y));
  }
}
*/

std::vector<Point2f> detect_landmard(string f1)
{
    try
    {

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        frontal_face_detector detector = get_frontal_face_detector();
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        deserialize("shape_predictor_68_face_landmarks.dat") >> sp;


        //image_window win, win_faces;
        // Loop over all the images provided on the command line.

	array2d<rgb_pixel> img;
	load_image(img, f1);
	// Make the image larger so we can detect small faces.
	pyramid_up(img);

	// Now tell the face detector to give us a list of bounding boxes
	// around all the faces in the image.
	std::vector<dlib::rectangle> dets = detector(img);
	cout << "Number of faces detected in " << f1 << ": " << dets.size() << endl;

	// Now we will go ask the shape_predictor to tell us the pose of
	// each face we detected.
	std::vector<Point2f> p1;
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		full_object_detection shape = sp(img, dets[j]);
		cout << "number of parts: "<< shape.num_parts() << endl;
		for (int x=0; x < shape.num_parts(); x++){
			float x_ = shape.part(x).x();
			float y_ = shape.part(x).y();
			p1.push_back(Point2f(x_,y_));
		}


	}
	return p1;


    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}



// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );

    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}


// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(Rect rect, std::vector<Point2f> &points, std::vector< std::vector<int> > &delaunayTri){

	// Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

	// Insert points into subdiv
    for( std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);

	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<Point2f> pt(3);
	std::vector<int> ind(3);

	for( size_t i = 0; i < triangleList.size(); i++ )
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5 ]);

		if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
			for(int j = 0; j < 3; j++)
				for(size_t k = 0; k < points.size(); k++)
					if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = k;

			delaunayTri.push_back(ind);
		}
	}

}


// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2)
{

    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // Offset points by left top corner of the respective rectangles
    std::vector<Point2f> t1Rect, t2Rect;
    std::vector<Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {

        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
        t2RectInt.push_back( Point(t2[i].x - r2.x, t2[i].y - r2.y) ); // for fillConvexPoly

    }

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    // Apply warpImage to small rectangular patches
    Mat img1Rect;
    img1(r1).copyTo(img1Rect);

    Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

    applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

    multiply(img2Rect,mask, img2Rect);
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2Rect;


}


int main( int argc, char** argv)
{
	//Read input images

    string filename1 = argv[1];
    string filename2 = argv[2];

    Mat img1 = imread(filename1);
    Mat img2 = imread(filename2);
    Mat img1Warped = img2.clone();

    //Read points
	std::vector<Point2f> points1, points2;
	points1 = detect_landmard(filename1);
	points2 = detect_landmard(filename2);

 for (int i = points1.begin(); points1.end(); i++){
	 cout << points1[x].x() << "," << points1[x].y() << endl;
 }

  int flg = 0;
	cout << flg++ << endl;
/*
	dlib::array2d<unsigned char> imgDlib1,imgDlib2;
  dlib::load_image(imgDlib1, argv[1]);
  dlib::load_image(imgDlib2, argv[2]);

  Mat imgCV1 = imread(argv[1]);
  Mat imgCV2 = imread(argv[2]);
shape_predictor sp;
  deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

  std::vector<Point2f> points1, points2;

  faceLandmarkDetection(imgDlib1, sp, points1);
  faceLandmarkDetection(imgDlib2, sp, points2);
    */
    //convert Mat to float data type
    img1.convertTo(img1, CV_32F);
    img1Warped.convertTo(img1Warped, CV_32F);
cout << flg++ << endl;

    // Find convex hull
    std::vector<Point2f> hull1;
    std::vector<Point2f> hull2;
    std::vector<int> hullIndex;

    convexHull(points2, hullIndex, false, false);
cout << flg++ << endl;
    for(int i = 0; i < hullIndex.size(); i++)
    {
        hull1.push_back(points1[hullIndex[i]]);
        hull2.push_back(points2[hullIndex[i]]);
    }
cout << flg++ << endl;

    // Find delaunay triangulation for points on the convex hull
    std::vector< std::vector<int> > dt;
	Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
	calculateDelaunayTriangles(rect, hull1, dt);
cout << flg++ << endl;
	// Apply affine transformation to Delaunay triangles
	for(size_t i = 0; i < dt.size(); i++)
    {
        std::vector<Point2f> t1, t2;
        // Get points for img1, img2 corresponding to the triangles
		for(size_t j = 0; j < 3; j++)
        {
			t1.push_back(hull1[dt[i][j]]);
			t2.push_back(hull2[dt[i][j]]);
		}

        warpTriangle(img1, img1Warped, t1, t2);

	}
cout << flg++ << endl;
    // Calculate mask
    std::vector<Point> hull8U;
    for(int i = 0; i < hull2.size(); i++)
    {
        Point pt(hull2[i].x, hull2[i].y);
        hull8U.push_back(pt);
    }

    Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
    fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,255,255));

    // Clone seamlessly.
    Rect r = boundingRect(hull2);
    Point center = (r.tl() + r.br()) / 2;

    Mat output;
    img1Warped.convertTo(img1Warped, CV_8UC3);
	seamlessClone(img1Warped,img2, mask, center, output, NORMAL_CLONE);

    imshow("Face Swapped", output);
    waitKey(0);
    destroyAllWindows();


	return 1;
}
