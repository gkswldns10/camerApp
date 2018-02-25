#include <stdio.h>
#include <vector>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <string.h>
#include <fstream>
#include <mat.hpp>


using namespace std;


#define PREDICTOR_PATH = "/Users/Kevin/Desktop/HackIllinois/camerApp/shape_predictor_68_face_landmarks.dat"
#define SCALE_FACTOR 1 
#define FEATHER_AMOUNT 11

static int FACE_POINTS[] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
	31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
	51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67};

static int MOUTH_POINTS[] = {17, 18, 19, 20, 21, 22};
static int RIGHT_BROW_POINTS[] = {22, 23, 24, 25, 26};
static int LEFT_BROW_POINTS[] = {36, 37, 38, 39, 40, 41, 42};
static int NOSE_POINTS = {27, 28, 29, 30, 31, 32, 33, 34};
static int JAW_POINTS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

void read_im(string fname){

	im = cv::imread(fname, cv::IMREAD_COLOR);
	im = cv::resize(im, (im.shape[1]*SCALE_FACTOR, im.shape[0]*SCALE_FACTOR));
}

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

int main(int argc, char** argv){
	

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(PREDICTOR_PATH)
}




















