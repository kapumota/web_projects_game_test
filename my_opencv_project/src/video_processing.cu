#include "video_processing.h"
#include <opencv2/opencv.hpp>
#include <cstdio>

using namespace cv;

// Función para procesar una secuencia de video
void testVideoSequence() {
    VideoCapture cap("Videos/rubic.avi"); // Ruta de tu video
    if (!cap.isOpened()) {
        printf("Cannot open video capture device.\n");
        return;
    }

    Mat edges, frame;
    char c;
    while (cap.read(frame)) {
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        Canny(grayFrame, edges, 40, 100, 3);
        imshow("source", frame);
        imshow("gray", grayFrame);
        imshow("edges", edges);
        c = waitKey(0);  // waits a key press to advance to the next frame
        if (c == 27) {
            printf("ESC pressed - capture finished\n");
            break;
        }
    }
}

// Función para capturar imágenes de video en vivo desde la cámara
void testSnap() {
    VideoCapture cap(0); // open the default camera (i.e., the built-in webcam)
    if (!cap.isOpened()) { // checking if the camera opened successfully
        printf("Cannot open video capture device.\n");
        return;
    }

    Mat frame;
    char numberStr[256];
    char fileName[256];

    // Video resolution
    Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                     (int)cap.get(CAP_PROP_FRAME_HEIGHT));

    // Display window for the source frame
    const char* WIN_SRC = "Source";
    namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
    moveWindow(WIN_SRC, 0, 0);

    // Display window for showing the snapped frame
    const char* WIN_DST = "Snapped";
    namedWindow(WIN_DST, WINDOW_AUTOSIZE);
    moveWindow(WIN_DST, capS.width + 10, 0);

    char c;
    int frameCount = 0;

    while (true) {
        cap >> frame; // capture a new frame from camera
        if (frame.empty()) {
            printf("End of the video capture\n");
            break;
        }

        imshow(WIN_SRC, frame);

        c = waitKey(10);  // waits for a key press to advance to the next frame
        if (c == 27) {  // press ESC to exit
            printf("ESC pressed - capture finished\n");
            break;
        }
        if (c == 115) {  //'s' pressed - snap the image to a file
            frameCount++;
            sprintf(numberStr, "%d", frameCount);
            strcpy(fileName, "Images/snap_");
            strcat(fileName, numberStr);
            strcat(fileName, ".bmp");
            bool bSuccess = imwrite(fileName, frame);
            if (!bSuccess) {
                printf("Error writing the snapped image\n");
            } else {
                imshow(WIN_DST, frame);
            }
        }
    }
}

