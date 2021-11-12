#include "Utilities.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <list>
#include <experimental/filesystem> // C++-standard header file name
#include <filesystem> // Microsoft-specific implementation header file name
using namespace std::experimental::filesystem::v1;
using namespace std;

#define FOCAL_LENGTH_ESTIMATE 1770
#define PLATE_WIDTH_IN_MM 465
#define PLATE_HEIGHT_IN_MM 100
#define FRAMES_PER_SECOND 29.97
#define REQUIRED_DICE 0.8
#define FRAME_WIDTH 960 // Added
#define FRAME_HEIGHT 540 // Added
#define FRAMES_FOR_DISTANCES_LEN 9 //Added
const int LICENCE_PLATE_LOCATIONS[][5] = { {1, 67, 88, 26, 6}, {2, 67, 88, 26, 6}, {3, 68, 88, 26, 6},
	{4, 69, 88, 26, 6}, {5, 70, 89, 26, 6}, {6, 70, 89, 27, 6}, {7, 71, 89, 27, 6}, {8, 73, 89, 27, 6},
	{9, 73, 90, 27, 6}, {10, 74, 90, 27, 6}, {11, 75, 90, 27, 6}, {12, 76, 90, 27, 6}, {13, 77, 91, 27, 6},
	{14, 78, 91, 27, 6}, {15, 78, 91, 27, 6}, {16, 79, 91, 27, 6}, {17, 80, 92, 27, 6}, {18, 81, 92, 27, 6},
	{19, 81, 92, 28, 6}, {20, 82, 93, 28, 6}, {21, 83, 93, 28, 6}, {22, 83, 93, 28, 6}, {23, 84, 93, 28, 6},
	{24, 85, 94, 28, 6}, {25, 85, 94, 28, 6}, {26, 86, 94, 28, 6}, {27, 86, 94, 28, 6}, {28, 86, 95, 29, 6},
	{29, 87, 95, 29, 6}, {30, 87, 95, 29, 6}, {31, 88, 95, 29, 6}, {32, 88, 96, 29, 6}, {33, 89, 96, 29, 6},
	{34, 89, 96, 29, 6}, {35, 89, 97, 29, 6}, {36, 90, 97, 29, 6}, {37, 90, 97, 30, 6}, {38, 91, 98, 30, 6},
	{39, 91, 98, 30, 6}, {40, 92, 98, 30, 7}, {41, 92, 99, 30, 7}, {42, 93, 99, 30, 7}, {43, 93, 99, 30, 7},
	{44, 94, 100, 30, 7}, {45, 95, 100, 30, 7}, {46, 95, 101, 30, 7}, {47, 96, 101, 30, 7}, {48, 97, 102, 30, 7},
	{49, 97, 102, 31, 7}, {50, 98, 102, 31, 7}, {51, 99, 103, 31, 7}, {52, 99, 103, 32, 7}, {53, 100, 104, 32, 7},
	{54, 101, 104, 32, 7}, {55, 102, 105, 32, 7}, {56, 103, 105, 32, 7}, {57, 104, 106, 32, 7}, {58, 105, 106, 32, 7},
	{59, 106, 107, 32, 7}, {60, 107, 107, 32, 7}, {61, 108, 108, 32, 7}, {62, 109, 108, 33, 7}, {63, 110, 109, 33, 7},
	{64, 111, 109, 33, 7}, {65, 112, 110, 34, 7}, {66, 113, 111, 34, 7}, {67, 114, 111, 34, 7}, {68, 116, 112, 34, 7},
	{69, 117, 112, 34, 8}, {70, 118, 113, 35, 8}, {71, 119, 113, 35, 8}, {72, 121, 114, 35, 8}, {73, 122, 114, 35, 8},
	{74, 124, 115, 35, 8}, {75, 125, 116, 36, 8}, {76, 127, 116, 36, 8}, {77, 128, 117, 36, 8}, {78, 130, 118, 36, 8},
	{79, 132, 118, 36, 9}, {80, 133, 119, 37, 9}, {81, 135, 120, 37, 9}, {82, 137, 121, 37, 9}, {83, 138, 122, 38, 9},
	{84, 140, 122, 38, 9}, {85, 142, 123, 38, 9}, {86, 144, 124, 38, 9}, {87, 146, 125, 38, 9}, {88, 148, 126, 39, 9},
	{89, 150, 127, 39, 9}, {90, 152, 128, 39, 9}, {91, 154, 129, 40, 9}, {92, 156, 129, 40, 10}, {93, 158, 130, 40, 10},
	{94, 160, 131, 41, 10}, {95, 163, 133, 41, 10}, {96, 165, 133, 41, 10}, {97, 167, 135, 42, 10}, {98, 170, 135, 42, 10},
	{99, 172, 137, 43, 10}, {100, 175, 138, 43, 10}, {101, 178, 139, 43, 10}, {102, 180, 140, 44, 10}, {103, 183, 141, 44, 10},
	{104, 186, 142, 44, 11}, {105, 188, 143, 45, 11}, {106, 192, 145, 45, 11}, {107, 195, 146, 45, 11}, {108, 198, 147, 45, 11},
	{109, 201, 149, 46, 11}, {110, 204, 150, 47, 11}, {111, 207, 151, 47, 11}, {112, 211, 152, 47, 11}, {113, 214, 154, 48, 11},
	{114, 218, 155, 48, 12}, {115, 221, 157, 49, 12}, {116, 225, 158, 50, 12}, {117, 229, 160, 50, 12}, {118, 234, 162, 50, 12},
	{119, 237, 163, 51, 12}, {120, 241, 164, 52, 12}, {121, 245, 166, 52, 12}, {122, 250, 168, 52, 12}, {123, 254, 169, 53, 12},
	{124, 258, 171, 54, 12}, {125, 263, 173, 55, 12}, {126, 268, 175, 55, 12}, {127, 273, 177, 55, 12}, {128, 278, 179, 56, 13},
	{129, 283, 181, 57, 13}, {130, 288, 183, 57, 13}, {131, 294, 185, 58, 13}, {132, 299, 187, 59, 13}, {133, 305, 190, 59, 13},
	{134, 311, 192, 60, 13}, {135, 317, 194, 60, 14}, {136, 324, 196, 60, 14}, {137, 330, 198, 61, 14}, {138, 336, 201, 63, 14},
	{139, 342, 203, 64, 14}, {140, 349, 206, 65, 14}, {141, 357, 208, 65, 15}, {142, 364, 211, 66, 15}, {143, 372, 214, 67, 15},
	{144, 379, 217, 68, 15}, {145, 387, 220, 69, 15}, {146, 396, 223, 70, 15}, {147, 404, 226, 71, 16}, {148, 412, 229, 72, 16},
	{149, 422, 232, 73, 17}, {150, 432, 236, 74, 17}, {151, 440, 239, 75, 18}, {152, 450, 243, 76, 18}, {153, 460, 247, 77, 18},
	{154, 470, 250, 78, 19}, {155, 482, 254, 78, 19}, {156, 492, 259, 81, 19}, {157, 504, 263, 82, 20}, {158, 516, 268, 83, 20},
	{159, 528, 272, 85, 21}, {160, 542, 277, 85, 21}, {161, 554, 282, 88, 21}, {162, 569, 287, 88, 22}, {163, 584, 292, 89, 22},
	{164, 598, 297, 91, 23}, {165, 614, 302, 92, 24}, {166, 630, 308, 94, 24}, {167, 646, 314, 96, 25}, {168, 664, 320, 97, 26},
	{169, 681, 327, 100, 26}, {170, 700, 334, 101, 27}, {171, 719, 341, 103, 28}, {172, 740, 349, 105, 29}, {173, 762, 357, 107, 29},
	{174, 784, 365, 109, 30}, { 175, 808, 374, 110, 31 }, { 176, 832, 383, 113, 32 } };
const int NUMBER_OF_PLATES = sizeof(LICENCE_PLATE_LOCATIONS) / (sizeof(LICENCE_PLATE_LOCATIONS[0]));
const int FRAMES_FOR_DISTANCES[] = { 54,   70,   86,  101,  115,  129,  143,  158,  172 };
const int DISTANCES_TRAVELLED_IN_MM[] = { 2380, 2380, 2400, 2380, 2395, 2380, 2385, 2380 };
const double SPEEDS_IN_KMPH[] = { 16.0, 16.0, 17.3, 18.3, 18.5, 18.3, 17.2, 18.3 };

float metric(int a, int b) {
	return (float)a / (float)(a + b);
}

bool hasLicensePlate(int frame_no) {
	return frame_no < NUMBER_OF_PLATES ? true : false;
}

int rectArea(int width, int height) {
	return width * height;
}

bool isIntersect(int* a_rect, int* b_rect) {
	if (a_rect[2] <= b_rect[0] || a_rect[0] >= b_rect[2]) return false;
	if (a_rect[1] >= b_rect[3] || a_rect[3] <= b_rect[1]) return false;
	return true;
}

int intersectionArea(Rect rect1, Rect rect2) {
	int ax1, ay1, ax2, ay2;
	int bx1, by1, bx2, by2;

	// a1 is left-bottom point
	ax1 = rect1.x;
	ay1 = rect1.y - rect1.height;
	// a2 is left-bottom point
	ax2 = rect1.x + rect1.width;
	ay2 = rect1.y;

	// b1 is left-bottom point
	bx1 = rect2.x;
	by1 = rect2.y - rect2.height;
	// b2 is left-bottom point
	bx2 = rect2.x + rect2.width;
	by2 = rect2.y;

	int a_rect[4] = { ax1, ay1, ax2, ay2 };
	int b_rect[4] = { bx1, by1, bx2, by2 };

	if (isIntersect(a_rect, b_rect)) {
		if (ax1 >= bx2 || ay1 >= by2 || ax2 <= bx1 || ay2 <= by1) {
			return (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1);
		}

		int rightX = min(ax2, bx2);
		int rightY = min(ay2, by2);
		int leftX = max(ax1, bx1);
		int leftY = max(ay1, by1);

		return (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - (rightX - leftX) * (rightY - leftY);
	}
	return 0;
}

float DICE(Rect rect1, Rect rect2) {
	int groundTruthArea = rectArea(rect1.width, rect1.height);
	int detectedArea = rectArea(rect2.width, rect2.height);
	int intersection = intersectionArea(rect1, rect2);
	int denominator = groundTruthArea + detectedArea;
	float ans = (float)(2 * intersection) / (float)denominator;
	return ans;

}

void drawPlate(Mat frame, Point2f* corners, cv::Scalar color) {
	cv::line(frame, corners[0], corners[1], color);
	cv::line(frame, corners[1], corners[2], color);
	cv::line(frame, corners[2], corners[3], color);
	cv::line(frame, corners[3], corners[0], color);
}

bool contains(const int frames[], int frame_number) {
	for (int i = 0; i < FRAMES_FOR_DISTANCES_LEN; i++) {
		if (frames[i] == frame_number) {
			return true;
		}
	}
	return false;
}

double distance_to_focus(int foundX, int foundWidth) {
	double h_camera = abs(foundX - (double)( FRAME_WIDTH / 2));
	double d_camera = hypot(h_camera, FOCAL_LENGTH_ESTIMATE);
	double d_mm = (double) (d_camera * PLATE_WIDTH_IN_MM) / (double)foundWidth;
	return d_mm;
}

double calculateAngle(double a, double b) {
	double angle = acos(a / b);
	return angle;
}

double calculateSide(double a, double b, double theta) {
	double c = sqrt(a*a + b*b - 2*a*b*cos(theta));
	return c;
}

double calculateDistance(Rect previous, Rect current) {
	double dmm1 = distance_to_focus(previous.x, previous.width);
	double radian1 = calculateAngle(FOCAL_LENGTH_ESTIMATE, dmm1);
	double dmm2 = distance_to_focus(current.x, current.width);
	double radian2 = calculateAngle(FOCAL_LENGTH_ESTIMATE, dmm2);
	double differenceAngel = abs(radian2 - radian1);
	double travelled = calculateSide(dmm1, dmm2, differenceAngel);
	return travelled;
}

double estimateSpeed(double distance, int frames) {
	double time = (double)frames / (double)FRAMES_PER_SECOND;
	double speed = (distance * 60 * 60) / (time * 1000 * 1000);
	return speed;
}

void applyGMM( Mat& src, Mat& dst, Mat& original_frame, Mat& structuring_element,
			   Ptr<BackgroundSubtractorMOG2> gmm) {
	Mat foreground_mask, thresholded_image;
	gmm->apply(src, foreground_mask);
	// Clean the resultant binary (moving pixel) mask using an opening.
	threshold(foreground_mask, thresholded_image, 150, 255, THRESH_BINARY);
	Mat closed_image, cleaned_foreground_mask;
	morphologyEx(thresholded_image, closed_image, MORPH_CLOSE, structuring_element);
	morphologyEx(closed_image, cleaned_foreground_mask, MORPH_OPEN, structuring_element);
	dst.setTo(Scalar(0, 0, 0));
	original_frame.copyTo(dst, cleaned_foreground_mask);
}

void otsuThresholding(Mat& src, Mat& dst) {
	Mat thresholded_image;
	cvtColor(src, thresholded_image, COLOR_BGR2GRAY);
	threshold(thresholded_image, dst, 20, 255, THRESH_BINARY | THRESH_OTSU);
}

void locateLicensePlate( Mat& cleaned_frame, Mat& found_object, Rect& object, 
						 Point2f location[], bool& objectLocated) {
	int rectIdx = -1;
	bool contourFound = false;
	double aspectRatio = 0;
	double biggestContourArea = 0;
	double width = 0;
	double height = 0;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Mat thresholded_image_copy = cleaned_frame.clone();
	findContours(thresholded_image_copy, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	
	for (int i = 0; i < contours.size(); i++) {
		RotatedRect rect = minAreaRect(contours[i]);
		double minVal = min(rect.size.width, rect.size.height);
		double maxVal = max(rect.size.width, rect.size.height);
		double tmpRatio = maxVal / minVal;
		double ctArea = cv::contourArea(contours[i]);

		if (tmpRatio >= 4.1 && tmpRatio < 4.95 && ctArea > biggestContourArea)
		{
			aspectRatio = tmpRatio;
			biggestContourArea = ctArea;
			width = maxVal;
			height = minVal;
			rectIdx = i;
			contourFound = true;
		}
	}

	objectLocated = contourFound;

	if (contourFound) {
		RotatedRect boundingBox = cv::minAreaRect(contours[rectIdx]);
		boundingBox.points(location);
		object = Rect(location[0].x, location[0].y, width, height);

		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		drawContours(found_object, contours, rectIdx, colour, cv::FILLED, 8, hierarchy);
	}
}

void updateConfusionMatrix( int confusion_matrix[], Rect givenRect, Rect foundRect,
							bool hasLicensePlate, bool objLocated) {
	float dice;
	// confusion_matrix[] 0=TP, 1=TN, 2=FP, 3=FN
	if (objLocated) {
		if (hasLicensePlate) {
			dice = DICE(givenRect, foundRect);
			if (dice > REQUIRED_DICE) {
				confusion_matrix[0]++;
			}
			else {
				confusion_matrix[3]++;
			}
		}
		else if (!hasLicensePlate) {
			confusion_matrix[2]++;
		}
	}
	else {
		if (hasLicensePlate) {
			confusion_matrix[3]++;
		}
		else {
			confusion_matrix[1]++;
		}
	}
}

void screenshot(int index, string img_prefix, Mat frame) {
	string index_name = to_string(index);
	string img_name = img_prefix + index_name + ".jpg";
	imwrite(img_name, frame);
}

void MyApplication()
{
	// TP, TN, FP, FN
	int TP = 0, TN = 0, FP = 0, FN = 0;
	int confusion_matrix[4] = { 0, 0, 0, 0 };

	int last_frame = 1;
	double dmm1 = 0, dmm2 = 0, angle1 = 0, angle2 = 0;

	string video_filename("Media/CarSpeedTest1.mp4");
	VideoCapture video;
	video.open(video_filename);

	string filename("Media/LicencePlateTemplate.png");
	Mat template_image = imread(filename, -1);
	string background_filename("Media/CarSpeedTest1EmptyFrame.jpg");
	Mat static_background_image = imread(background_filename, -1);
	if ((!video.isOpened()) || (template_image.empty()) || (static_background_image.empty()))
	{
		if (!video.isOpened())
			cout << "Cannot open video file: " << video_filename << endl;
		if (template_image.empty())
			cout << "Cannot open image file: " << filename << endl;
		if (static_background_image.empty())
			cout << "Cannot open image file: " << background_filename << endl;
	}
	else
	{
		Mat gray_template_image;
		cvtColor(template_image, gray_template_image, COLOR_BGR2GRAY);

		Mat current_frame;
		Mat difference_frame, binary_difference, thresholded_image, thresholded_image_2;
		video.set(cv::CAP_PROP_POS_FRAMES, 1);
		video >> current_frame;
		int frame_number = 1;
		double last_time = static_cast<double>(getTickCount());
		double frame_rate = video.get(cv::CAP_PROP_FPS);
		double time_between_frames = 1000.0 / frame_rate;

		Ptr<BackgroundSubtractorMOG2> gmm = createBackgroundSubtractorMOG2();
		Mat foreground_mask, foreground_image;
		Mat structuring_element(1, 1, CV_8U, Scalar(1));

		Rect previous(0,0,0,0);

		while (!current_frame.empty()) {
			char frame_no[20];
			Mat process_frame, gmm_output_frame, cleaned_frame;

			// Lower the brightness
			current_frame.convertTo(process_frame, -1, 1, -55);

			// GMM
			applyGMM(process_frame, gmm_output_frame, current_frame, structuring_element, gmm);

			// Cleaning the image by removing the noise
			otsuThresholding(gmm_output_frame, cleaned_frame);

			Point2f location[4];
			Rect foundObj;
			bool objLocated;
			bool isPlateExist = hasLicensePlate(frame_number);
			Mat contour_image = Mat::zeros(current_frame.size(), CV_8UC3);
			locateLicensePlate(cleaned_frame, contour_image, foundObj, location, objLocated);

			const int* givenData = LICENCE_PLATE_LOCATIONS[frame_number];
			Rect groudTruth(givenData[1], givenData[2], givenData[3], givenData[4]);

			//cout << "Frame No:" << frame_number;
			updateConfusionMatrix(confusion_matrix, groudTruth, foundObj, isPlateExist, objLocated);
			
			if (objLocated) {
				cv::Scalar white = Scalar(255, 255, 255);
				drawPlate(current_frame, location, white);

				// Speed calculation
				if (previous.width == 0) {
					previous = foundObj;
					last_frame = frame_number;
				}
				else if (previous.width != 0) {
					int frame_diff = frame_number - last_frame;

					double distance = calculateDistance(previous, foundObj);
					double speed = estimateSpeed(distance, frame_diff);

					if (contains(FRAMES_FOR_DISTANCES, frame_number)) {
						cout << "Frame "<< frame_number << " distance travelled "<< distance << " with speed " << speed << "\n";
					}

					string s = std::to_string(speed);
					string speed_text = "Speed: " + s + "kmph";
					Point speed_text_location(foundObj.x, foundObj.y - foundObj.height);
					Scalar speed_text_colour(0xFF, 0xFF, 0xFF);
					putText(current_frame, speed_text, speed_text_location, FONT_HERSHEY_SIMPLEX, 0.5, speed_text_colour);
					
					previous = foundObj;
					last_frame = frame_number;
				}
			}

				sprintf(frame_no, "%d", frame_number);
				Point frame_no_location(5, 15);
				Scalar frame_no_colour(0, 0, 0xFF);
				putText(current_frame, frame_no, frame_no_location, FONT_HERSHEY_SIMPLEX, 0.4, frame_no_colour);

				// Draw the ground truth plate
				int adjusted_frame = frame_number-1;
				cv::Point2f groundCorners[4];
				cv::Scalar blue = Scalar(0, 0, 255);
				groundCorners[0] = cv::Point2f(LICENCE_PLATE_LOCATIONS[adjusted_frame][1], LICENCE_PLATE_LOCATIONS[adjusted_frame][2]);
				groundCorners[1] = cv::Point2f(LICENCE_PLATE_LOCATIONS[adjusted_frame][1] + LICENCE_PLATE_LOCATIONS[adjusted_frame][3], LICENCE_PLATE_LOCATIONS[adjusted_frame][2]);
				groundCorners[2] = cv::Point2f(LICENCE_PLATE_LOCATIONS[adjusted_frame][1] + LICENCE_PLATE_LOCATIONS[adjusted_frame][3], LICENCE_PLATE_LOCATIONS[adjusted_frame][2] + LICENCE_PLATE_LOCATIONS[adjusted_frame][4]);
				groundCorners[3] = cv::Point2f(LICENCE_PLATE_LOCATIONS[adjusted_frame][1], LICENCE_PLATE_LOCATIONS[adjusted_frame][2] + LICENCE_PLATE_LOCATIONS[adjusted_frame][4]);
				drawPlate(current_frame, groundCorners, blue);

				cv::imshow("Processed video", current_frame);

				double current_time = static_cast<double>(getTickCount());
				double duration = (current_time - last_time) / getTickFrequency() / 1000.0;
				int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
				last_time = current_time;
				char c = cv::waitKey(delay);
				for (int i = 0; i < 1; i++)
					video >> current_frame;
				frame_number++;
			}
			cv::destroyAllWindows();

			// TP, TN, FP, FN
			float recall = metric(confusion_matrix[0], confusion_matrix[3]);
			float precision = metric(confusion_matrix[0], confusion_matrix[2]);
			cout << "Result:\nTP: " << confusion_matrix[0] << " TN: " << confusion_matrix[1] << " FP: " << confusion_matrix[2] << " FN: " << confusion_matrix[3];
			cout << "\nRecall: " << recall << "\nPrecision: " << precision;
		}
}


