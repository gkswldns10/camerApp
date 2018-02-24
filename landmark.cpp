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


  dlib::array2d<unsigned char> imgDlib1,imgDlib2;
  dlib::load_image(imgDlib1, argv[1]);
  dlib::load_image(imgDlib2, argv[2]);

  Mat imgCV1 = imread(argv[1]);
  Mat imgCV2 = imread(argv[2]);
  if(!imgCV1.data || !imgCV2.data)
  {
    printf("No image data \n");
    return -1;
  }

  //---------------------- step 2. detect face landmarks -----------------------------------
  shape_predictor sp;
  deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

  std::vector<Point2f> points1, points2;

  faceLandmarkDetection(imgDlib1, sp, points1);
  faceLandmarkDetection(imgDlib2, sp, points2);
  

  //---------------------step 3. find convex hull -------------------------------------------
  Mat imgCV1Warped = imgCV2.clone();
  imgCV1.convertTo(imgCV1, CV_32F);
  imgCV1Warped.convertTo(imgCV1Warped, CV_32F);
  if(v) time_teack(&start, "convertWarped");
#ifdef DEBUG
  draw_face("Face 1" ,imgCV1, points1);
#endif
if(v) time_teack(&start, "faceLandmarkDetection2");
#ifdef DEBUG
  draw_face("Face 2", imgCV2, points2);
#endif
