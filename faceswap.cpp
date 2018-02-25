#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image_abstract.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <dlib/gui_widgets.h>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace dlib;
using namespace std;

//Read points from text file
std::vector<std::vector<Point2f>> readPoints(string pointsFileName){
std::vector<Point2f> points;
	std::vector<std::vector<Point2f>> set_pts;
	ifstream ifs (pointsFileName.c_str());
    float x, y;
	int count = 1;
	int idx = 1;
    while(ifs >> x >> y)
    {
				if (count == 68*idx){
					set_pts.push_back(points);
					std::vector<Point2f> points;
					idx++;
				}
        points.push_back(Point2f(x,y));
				count++;
    }

	return set_pts;
}


//Read points from get_frontal_face_detector
std::vector<std::vector<Point2f>> detect_landmark(string f1)
{
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor sp;
	dlib::array2d<unsigned char> imgDlib;
  deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
  dlib::load_image(imgDlib, f1);
	//pyramid_up(imgDlib);
  //dlib::pyramid_up(imgDlib);
  std::vector<dlib::rectangle> dets = detector(imgDlib);
  //DLOG("Number of faces detected: %lu \n", dets.size());

 std::vector<std::vector<Point2f>> set_pts;
  for (unsigned long j = 0; j < dets.size(); ++j)
	{
    full_object_detection shape = sp(imgDlib, dets[j]);
		std::vector<Point2f> points;
    for (int i = 0; i < shape.num_parts(); ++i) {
      float x=shape.part(i).x();
      float y=shape.part(i).y();
      points.push_back(Point2f(x,y));
    }
		set_pts.push_back(points);
  }
  return set_pts;
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
// Returns the std::vector of indices of 3 points for each triangle
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


// int main( int argc, char** argv)
// {
// 	//Read input images
//     string filename1 = argv[1];
//     string filename2 = argv[2];
//
//     Mat img1 = imread(filename1);
//     Mat img2 = imread(filename2);
//     Mat img1Warped = img2.clone();
// 	/*
// 	txt txt_1 = txt(filename1);
// 	txt txt_2 = txt(filename2);
//
// 	txt_1.make_txt();
// 	txt_2.make_txt();
//     */
//     //Read points
// 	std::vector<std::vector<Point2f>> set_pts1, set_pts2;
// 	set_pts1 = detect_landmark(filename1);
// 	set_pts2 = detect_landmark(filename2);
// 	for (int abc = 0; abc < set_pts1.size(); abc++){
//
// 		std::vector<Point2f> points1 = set_pts1[abc];
// 		std::vector<Point2f> points2 = set_pts2[abc];
//
//     //convert Mat to float data type
//     img1.convertTo(img1, CV_32F);
//     img1Warped.convertTo(img1Warped, CV_32F);
//     cout << set_pts1.size() << endl;
//
//     // Find convex hull
//     std::vector<Point2f> hull1;
//     std::vector<Point2f> hull2;
//     std::vector<int> hullIndex;
//
//     convexHull(points2, hullIndex, false, false);
//
//     for(int i = 0; i < hullIndex.size(); i++)
//     {
//         hull1.push_back(points1[hullIndex[i]]);
//         hull2.push_back(points2[hullIndex[i]]);
//     }
//
//
// 	    // Find delaunay triangulation for points on the convex hull
// 	    std::vector< std::vector<int> > dt;
// 		Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
// 		calculateDelaunayTriangles(rect, hull2, dt);
//
// 	// Apply affine transformation to Delaunay triangles
// 	for(size_t i = 0; i < dt.size(); i++)
//     {
//         std::vector<Point2f> t1, t2;
//         // Get points for img1, img2 corresponding to the triangles
// 		for(size_t j = 0; j < 3; j++)
//         {
// 			t1.push_back(hull1[dt[i][j]]);
// 			t2.push_back(hull2[dt[i][j]]);
// 		}
//
//         warpTriangle(img1, img1Warped, t1, t2);
//
// 	}
//
//     // Calculate mask
//     std::vector<Point> hull8U;
//     for(int i = 0; i < hull2.size(); i++)
//     {
//         Point pt(hull2[i].x, hull2[i].y);
//         hull8U.push_back(pt);
//     }
//
//     Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
//     fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,255,255));
//
//     // Clone seamlessly.
//     Rect r = boundingRect(hull2);
//     Point center = (r.tl() + r.br()) / 2;
//
//     Mat output;
//     img1Warped.convertTo(img1Warped, CV_8UC3);
// 	seamlessClone(img1Warped,img2, mask, center, output, NORMAL_CLONE);
//
//
//     imshow("Face Swapped", output);
//     waitKey(0);
//     destroyAllWindows();
// }
//
// 	return 1;
// }

void two_swap(string filename1, std::vector<Point2f> points1, string filename2, std::vector<Point2f> points2)
{


	    Mat img1 = imread(filename1);
	    Mat img2 = imread(filename2);
	    Mat img1Warped = img2.clone();
			//convert Mat to float data type
			img1.convertTo(img1, CV_32F);
			img1Warped.convertTo(img1Warped, CV_32F);

	    // Find convex hull
	    std::vector<Point2f> hull1;
	    std::vector<Point2f> hull2;
	    std::vector<int> hullIndex;
	    convexHull(points2, hullIndex, false, false);

	    for(int i = 0; i < hullIndex.size(); i++)
	    {
	        hull1.push_back(points1[hullIndex[i]]);
	        hull2.push_back(points2[hullIndex[i]]);
	    }


		    // Find delaunay triangulation for points on the convex hull
		    std::vector< std::vector<int> > dt;
			Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
			calculateDelaunayTriangles(rect, hull2, dt);

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
}


int main( int argc, char** argv)
{
	//Read input images
    string filename1 = argv[1];
    string filename2 = argv[2];


    //Read points
		std::vector<std::vector<Point2f>> set_pts1, set_pts2;
		set_pts1 = detect_landmark(filename1);
		cout << "image 1 done" << endl;
		set_pts2 = detect_landmark(filename2);
		cout << "image 2 done" << endl;

		int abc1, abc2;
		cout << "Enter image to be replaced (1 ~ " << set_pts1.size() << "): ";
		cin >> abc1;
		cout << "Enter image to replace with (1 ~ " << set_pts1.size() << "): ";
		cin >> abc2;



		two_swap(filename1, set_pts1[abc1-1], filename2, set_pts2[abc2-1]);

	return 1;
}
