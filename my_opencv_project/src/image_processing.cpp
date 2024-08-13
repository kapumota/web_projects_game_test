#include "image_processing.h"
#include <string>  // Para usar std::string
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>  // Para manejar directorios en sistemas Unix

#define WEAK 128
#define STRONG 255

using namespace cv;

// Función para abrir y guardar una imagen
void testOpenImage(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath);
    if (!src.empty()) {
        std::string outputFilePath = std::string(outputFolder) + "/open_image.jpg";
        imwrite(outputFilePath, src); // Guardar la imagen
    } else {
        printf("Error: Could not open or find the image.\n");
    }
}

// Función para convertir una imagen a su negativo y guardarla
void testNegativeImage(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath, IMREAD_GRAYSCALE);
    if (src.empty()) {
        printf("Error: Could not open or find the image.\n");
        return;
    }

    int height = src.rows;
    int width = src.cols;
    Mat dst = Mat(height, width, CV_8UC1);

    // Generar imagen negativa
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uchar val = src.at<uchar>(i, j);
            uchar neg = 255 - val;
            dst.at<uchar>(i, j) = neg;
        }
    }

    std::string outputFilePath = std::string(outputFolder) + "/negative_image.jpg";
    imwrite(outputFilePath, dst); // Guardar la imagen
}

// Implementación de Canny con Hough transform y guardar el resultado
void metodaCannyProiect(Mat src, int kElemente, const char* outputFolder) {
    int hough[700][700] = {0};  // Declara hough como una variable local
    Mat temp = src.clone();
    Mat modul = Mat::zeros(src.size(), CV_8UC1);
    Mat directie = Mat::zeros(src.size(), CV_8UC1);

    int Sx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Sy[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    temp = filtruGaussianProiect(src);
    float gradX = 0;
    float gradY = 0;
    int k = 1;

    for (int y = k; y < temp.rows - k; y++) {
        for (int x = k; y < temp.cols - k; x++) {
            int auxX = 0;
            int auxY = 0;
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
                    auxX += temp.at<uchar>(y + i, x + j) * Sx[i + k][j + k];
                    auxY += temp.at<uchar>(y + i, x + j) * Sy[i + k][j + k];
                }
            }

            gradX = (float)auxX;
            gradY = (float)auxY;

            modul.at<uchar>(y, x) = (uchar)(sqrt(gradX * gradX + gradY * gradY) / 5.65);

            int dir = 0;
            float teta = atan2((float)gradY, (float)gradX);
            if ((teta > 3 * CV_PI / 8 && teta < 5 * CV_PI / 8) || (teta > -5 * CV_PI / 8 && teta < -3 * CV_PI / 8)) {
                dir = 0;
            } else if ((teta > CV_PI / 8 && teta < 3 * CV_PI / 8) || (teta > -7 * CV_PI / 8 && teta < -5 * CV_PI / 8)) {
                dir = 1;
            } else if ((teta > -CV_PI / 8 && teta < CV_PI / 8) || teta > 7 * CV_PI / 8 && teta < -7 * CV_PI / 8) {
                dir = 2;
            } else if ((teta > 5 * CV_PI / 8 && teta < 7 * CV_PI / 8) || (teta > -3 * CV_PI / 8 && teta < -CV_PI / 8)) {
                dir = 3;
            }
            directie.at<uchar>(y, x) = dir;
        }
    }

    // Supresión de no máximos
    for (int i = 1; i < modul.rows - 1; i++) {
        for (int j = 1; j < modul.cols - 1; j++) {
            if (directie.at<uchar>(i, j) == 0) {
                if ((modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j)) || (modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j)))
                    modul.at<uchar>(i, j) = 0;
            } else if (directie.at<uchar>(i, j) == 1) {
                if ((modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j + 1)) || (modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j - 1)))
                    modul.at<uchar>(i, j) = 0;
            } else if (directie.at<uchar>(i, j) == 2) {
                if ((modul.at<uchar>(i, j) < modul.at<uchar>(i, j - 1)) || (modul.at<uchar>(i, j) < modul.at<uchar>(i, j + 1)))
                    modul.at<uchar>(i, j) = 0;
            } else if (directie.at<uchar>(i, j) == 3) {
                if ((modul.at<uchar>(i, j) < modul.at<uchar>(i - 1, j - 1)) || (modul.at<uchar>(i, j) < modul.at<uchar>(i + 1, j + 1)))
                    modul.at<uchar>(i, j) = 0;
            }
        }
    }

    // Preprocesamiento de histeresis para detección de bordes
    float p = 0.08f;
    float K = 0.4f;

    int histograma[256] = { 0 };
    for (int i = 0; i < modul.rows; i++) {
        for (int j = 0; j < modul.cols; j++) {
            histograma[modul.at<uchar>(i, j)]++;
        }
    }

    float nrNonMuchie = (1 - p) * (modul.rows * modul.cols - histograma[0]);
    float suma = 0;
    float pragAdaptiv = 0;
    for (int i = 1; i < 256; i++) {
        suma += histograma[i];
        if (suma > nrNonMuchie) {
            pragAdaptiv = (float)i;
            break;
        }
    }

    float pragInalt = pragAdaptiv;
    float pragJos = K * pragInalt;

    for (int i = 1; i < modul.rows - 1; i++) {
        for (int j = 1; j < modul.cols - 1; j++) {
            if (modul.at<uchar>(i, j) < pragJos) {
                modul.at<uchar>(i, j) = 0;
            } else if (modul.at<uchar>(i, j) > pragInalt) {
                modul.at<uchar>(i, j) = STRONG;
            } else if ((pragJos < modul.at<uchar>(i, j)) && (modul.at<uchar>(i, j) < pragInalt)) {
                modul.at<uchar>(i, j) = WEAK;
            }
        }
    }

    Mat destinatie;
    cvtColor(modul, destinatie, COLOR_GRAY2RGB);

    struct Max {
        int ii, jj, val;
    };

    struct Max maximStructura[5000];
    int n = 0;

    for (int i = 0; i < modul.rows; i++) {
        for (int j = 0; j < modul.cols; j++) {
            int curent = hough[i][j];
            int afirmativ = 0;

            for (int x = 0; x < 7; x++) {
                for (int y = 0; y < 7; y++) {
                    if (hough[i + x][j + y] > curent) {
                        afirmativ++;
                    }
                }
            }

            if ((afirmativ == 0) && (hough[i][j] >= 10)) {
                maximStructura[n].ii = i;
                maximStructura[n].jj = j;
                maximStructura[n].val = hough[i][j];
                n++;
            }
        }
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (maximStructura[i].val < maximStructura[j].val) {
                Max aux = maximStructura[i];
                maximStructura[i] = maximStructura[j];
                maximStructura[j] = aux;
            }
        }
    }

    for (int n = 0; n < kElemente; n++) {
        int p = maximStructura[n].ii;

        for (int theta = 0; theta < 360; theta++) {
            for (int i = 0; i < destinatie.rows; i++) {
                for (int j = 0; j < destinatie.cols; j++) {
                    if (((float)i * cos(theta) + (float)j * sin(theta)) == p) {
                        destinatie.at<Vec3b>(i, j) = Vec3b(255, 0, 255); // Magenta
                    }
                }
            }
        }
    }

    std::string outputFilePath = std::string(outputFolder) + "/hough_transform.jpg";
    imwrite(outputFilePath, destinatie); // Guardar la imagen
}

// Función para aplicar un filtro Gaussiano a una imagen y guardarla
void testBlurImage(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath);
    if (src.empty()) {
        printf("Error: Could not open or find the image.\n");
        return;
    }
    Mat blurred;
    GaussianBlur(src, blurred, Size(15, 15), 0);
    
    std::string outputFilePath = std::string(outputFolder) + "/blurred_image.jpg";
    imwrite(outputFilePath, blurred); // Guardar la imagen
}

// Función para aplicar la detección de bordes usando Sobel y guardarla
void testEdgeDetection(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath, IMREAD_GRAYSCALE);
    if (src.empty()) {
        printf("Error: Could not open or find the image.\n");
        return;
    }

    Mat grad_x, grad_y, edges;
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;

    Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    Mat abs_grad_x, abs_grad_y;
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);

    std::string outputFilePath = std::string(outputFolder) + "/edge_detection.jpg";
    imwrite(outputFilePath, edges); // Guardar la imagen
}

// Función para aplicar la inversión de colores a la imagen y guardarla
void testParcurgereSimplaDiblookStyle(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath);
    if (src.empty()) {
        printf("Error: Could not open or find the image.\n");
        return;
    }

    // Ejemplo de recorrido simple para manipular cada píxel de la imagen
    Mat dst = src.clone();
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b color = src.at<Vec3b>(Point(x, y));

            // Invertir los colores de la imagen
            color[0] = 255 - color[0];  // B
            color[1] = 255 - color[1];  // G
            color[2] = 255 - color[2];  // R

            dst.at<Vec3b>(Point(x, y)) = color;
        }
    }

    std::string outputFilePath = std::string(outputFolder) + "/inverted_colors.jpg";
    imwrite(outputFilePath, dst); // Guardar la imagen
}

// Función para convertir una imagen BGR a HSV y guardarla
void testBGR2HSV(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath);
    if (src.empty()) {
        printf("Error: Could not open or find the image.\n");
        return;
    }

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    std::string outputFilePath = std::string(outputFolder) + "/hsv_image.jpg";
    imwrite(outputFilePath, hsv); // Guardar la imagen
}

// Función para redimensionar una imagen y guardarla
void testResize(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath);
    if (src.empty()) {
        printf("Error: Could not open or find the image.\n");
        return;
    }

    Mat resized;
    resize(src, resized, Size(src.cols / 2, src.rows / 2));  // Redimensionar a la mitad

    std::string outputFilePath = std::string(outputFolder) + "/resized_image.jpg";
    imwrite(outputFilePath, resized); // Guardar la imagen
}

// Función para aplicar Canny edge detection y guardarla
void testCanny(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath);
    if (src.empty()) {
        printf("Error: Could not open or find the image.\n");
        return;
    }

    Mat edges;
    Canny(src, edges, 100, 200);

    std::string outputFilePath = std::string(outputFolder) + "/canny_edges.jpg";
    imwrite(outputFilePath, edges); // Guardar la imagen
}

// Implementación simple de procesamiento de imagen y guardarla
void proiect(const char* imagePath, const char* outputFolder) {
    Mat src = imread(imagePath);
    if (src.empty()) {
        printf("Error: Could not open or find the image.\n");
        return;
    }

    // Procesamiento de imagen
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    std::string outputFilePath = std::string(outputFolder) + "/grayscale_image.jpg";
    imwrite(outputFilePath, gray); // Guardar la imagen
}

// Función para aplicar un filtro Gaussiano
cv::Mat filtruGaussianProiect(cv::Mat src) {
    Mat dst = src.clone();
    int w = 5;
    float sigma = (float)w / 6;
    int k = w / 2;
    float G[5][5] = { 0 };

    float suma = 0.0;
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < w; y++) {
            G[x][y] = (float)(1.0 / (2 * CV_PI * sigma * sigma) * exp(-(((x - 2) * (x - 2) + (y - 2) * (y - 2)) / (2 * sigma * sigma))));
            suma += G[x][y];
        }
    }

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < w; y++) {
            G[x][y] /= suma;
        }
    }

    for (int x = k; x < src.rows - k; x++) {
        for (int y = k; y < src.cols - k; y++) {
            int sum = 0;
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
                    sum += (int)(src.at<uchar>(x + i, y + j) * G[i + k][j + k]);
                }
            }
            dst.at<uchar>(x, y) = sum;
        }
    }
    return dst;
}


