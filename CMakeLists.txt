cmake_minimum_required(VERSION 3.10)
project(spectrogram)

find_package(SndFile REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(spectrogram main.cpp)
target_link_libraries(spectrogram
    sndfile
    fftw3
    ${OpenCV_LIBS}
)
