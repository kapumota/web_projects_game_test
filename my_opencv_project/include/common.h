#ifndef COMMON_H
#define COMMON_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <queue>

using namespace cv;
using namespace std;

#define WEAK 128
#define STRONG 255

bool openFileDlg(char* fname);
bool openFolderDlg(char* folderName);
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height);

#endif // COMMON_H


/*
#ifndef COMMON_H
#define COMMON_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <queue>

using namespace cv;
using namespace std;

#define WEAK 128
#define STRONG 255

bool openFileDlg(char* fname);
bool openFolderDlg(char* folderName);
void showHistogram(const std::string& name, int* hist, const int hist_cols, const int hist_height);

// Nueva función que usará CUDA para realizar operaciones paralelizadas
void commonFunction(int* data, int size);

#endif // COMMON_H


*/
