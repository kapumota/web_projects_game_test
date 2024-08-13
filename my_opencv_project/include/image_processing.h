#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

void testOpenImage(const char* imagePath, const char* outputFolder);
void testParcurgereSimplaDiblookStyle(const char* imagePath, const char* outputFolder);
void testBGR2HSV(const char* imagePath, const char* outputFolder);
void testResize(const char* imagePath, const char* outputFolder);
void testCanny(const char* imagePath, const char* outputFolder);
void testBlurImage(const char* imagePath, const char* outputFolder);
void testEdgeDetection(const char* imagePath, const char* outputFolder);
void proiect(const char* imagePath, const char* outputFolder);
cv::Mat filtruGaussianProiect(cv::Mat src);  // Agrega esta línea
void metodaCannyProiect(cv::Mat src, int kElemente, const char* outputFolder);

#endif // IMAGE_PROCESSING_H

/*

#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

void testOpenImage(const char* imagePath, const char* outputFolder);
void testParcurgereSimplaDiblookStyle(const char* imagePath, const char* outputFolder);
void testBGR2HSV(const char* imagePath, const char* outputFolder);
void testResize(const char* imagePath, const char* outputFolder);
void testCanny(const char* imagePath, const char* outputFolder);
void testBlurImage(const char* imagePath, const char* outputFolder);
void testEdgeDetection(const char* imagePath, const char* outputFolder);
void proiect(const char* imagePath, const char* outputFolder);
cv::Mat filtruGaussianProiect(cv::Mat src);
void metodaCannyProiect(cv::Mat src, int kElemente, const char* outputFolder);

// Declaraciones para funciones CUDA
void processImage(const cv::Mat& input, cv::Mat& output);

#endif // IMAGE_PROCESSING_H

*/

