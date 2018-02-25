#ifndef __TXT_H__
#define __TXT_H__

#include <vector>
#include <iostream>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image_abstract.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace cv;
using namespace dlib;
using namespace std;

class txt{
	public:
		txt(string fname);		
		dlib::array2d<unsigned char> imgDlib;
	  	//dlib::load_image(imgDlib2, filename2);
		void make_txt();//, std::vector<Point2f>& landmark);
	private:
		string filename;
};

#endif

