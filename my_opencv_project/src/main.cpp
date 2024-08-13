#include "common.h"
#include "image_processing.h"
#include "video_processing.h"
#include <sys/stat.h>
#include <sys/types.h>

// Función para crear una carpeta si no existe
void createResultsFolder(const char* folderPath) {
    struct stat st = {0};

    if (stat(folderPath, &st) == -1) {
        mkdir(folderPath, 0700);
    }
}

int main() {
    // Ruta de la imagen lena.jpg
    const char* imagePath = "/home/c-lara/Descargas/my_opencv_project/Images/lena.jpg";
    
    // Carpeta de resultados
    const char* outputFolder = "/home/c-lara/Descargas/my_opencv_project/Results";
    createResultsFolder(outputFolder);  // Crear la carpeta si no existe

    // Abre y procesa la imagen con cada opción
    printf("Opening and processing image %s with different filters and transformations.\n", imagePath);

    // 1 - Open image
    printf("Option 1: Open image\n");
    testOpenImage(imagePath, outputFolder);

    // 3 - Image negative - diblook style
    printf("Option 3: Image negative - diblook style\n");
    testParcurgereSimplaDiblookStyle(imagePath, outputFolder);

    // 4 - BGR->HSV
    printf("Option 4: BGR->HSV\n");
    testBGR2HSV(imagePath, outputFolder);

    // 5 - Resize image
    printf("Option 5: Resize image\n");
    testResize(imagePath, outputFolder);

    // 6 - Canny edge detection
    printf("Option 6: Canny edge detection\n");
    testCanny(imagePath, outputFolder);

    // 10 - Blur Image
    printf("Option 10: Blur Image\n");
    testBlurImage(imagePath, outputFolder);

    // 11 - Edge Detection
    printf("Option 11: Edge Detection\n");
    testEdgeDetection(imagePath, outputFolder);

    // 30 - Proiect (Este es un ejemplo simple de procesamiento de imagen)
    printf("Option 30: Proiect (simple image processing)\n");
    proiect(imagePath, outputFolder);

    printf("All operations completed. Results saved in %s. Press any key to exit.\n", outputFolder);
    waitKey(0);
    return 0;
}


