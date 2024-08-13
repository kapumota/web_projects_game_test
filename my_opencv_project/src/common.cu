#include "common.h"
#include <dirent.h>
#include <cstring>
#include <opencv2/opencv.hpp>

using namespace cv;

// Function to open a file dialog and get the file name
bool openFileDlg(char* fname) {
    printf("Enter the path to the image file: ");
    scanf("%s", fname);

    // Check if the file exists
    FILE *file;
    if ((file = fopen(fname, "r"))) {
        fclose(file);
        return true;
    } else {
        printf("File does not exist.
");
        return false;
    }
}

// Function to open a folder dialog and get the folder name
bool openFolderDlg(char* folderName) {
    printf("Enter the path to the folder: ");
    scanf("%s", folderName);

    // Check if the folder exists
    DIR *dir = opendir(folderName);
    if (dir) {
        closedir(dir);
        return true;
    } else {
        printf("Folder does not exist.
");
        return false;
    }
}

// Function to display a histogram
void showHistogram(const std::string& name, int* hist, const int hist_cols, const int hist_height) {
    // Create an image to display the histogram
    Mat imgHist(hist_height, hist_cols, CV_8UC3, Scalar(255, 255, 255)); // White background

    // Find the maximum value in the histogram
    int max_hist = 0;
    for (int i = 0; i < hist_cols; i++) {
        if (hist[i] > max_hist) {
            max_hist = hist[i];
        }
    }

    // Scale factor to fit the histogram in the image
    double scale = (double)hist_height / max_hist;
    int baseline = hist_height - 1;

    // Draw the histogram
    for (int x = 0; x < hist_cols; x++) {
        Point p1 = Point(x, baseline);
        Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
        line(imgHist, p1, p2, Scalar(255, 0, 255), 1); // Magenta colored lines
    }

    // Show the histogram image
    imshow(name, imgHist);
    waitKey(0); // Wait for a key press to close the window
}
