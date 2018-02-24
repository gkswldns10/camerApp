#include "txt.hpp"

txt::txt(string fname){
  filename = fname;
}

void txt::make_txt() //, std::vector<Point2f>& landmark)
{
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor sp;
  deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
  dlib::load_image(imgDlib, filename);
//  dlib::pyramid_up(img);
  ofstream outputFile;
  outputFile.open(filename + ".txt");

  std::vector<dlib::rectangle> dets = detector(imgDlib);
  //DLOG("Number of faces detected: %lu \n", dets.size());

  full_object_detection shape = sp(imgDlib, dets[0]);

  for (int i = 0; i < shape.num_parts(); ++i) {
    float x=shape.part(i).x();
    float y=shape.part(i).y();
    outputFile << x << " " << y << endl;
    //landmark.push_back(Point2f(x,y));
  }
  outputFile.close();
}

int main(){
  txt txt1 = txt("1.jpg");
  txt txt2 = txt("2.jpg");
  txt1.make_txt();
  txt2.make_txt();
  return 0;
}
