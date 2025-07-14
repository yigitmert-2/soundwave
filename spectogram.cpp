#include <iostream>
#include <filesystem>
#include <sndfile.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input.wav\n";
        return 1;
    }

    //Read audio into a float buffer
    SF_INFO sfinfo;
    SNDFILE* snd = sf_open(argv[1], SFM_READ, &sfinfo);
    if (!snd) { perror("sf_open"); return 1; }
    std::vector<float> audio(sfinfo.frames * sfinfo.channels);
    sf_readf_float(snd, audio.data(), sfinfo.frames);
    sf_close(snd);

    //STFT parameters
    const int chan      = sfinfo.channels;
    const int NFFT      = 1024;
    const int HOP       = NFFT / 4;
    const int freqBins  = NFFT / 2 + 1;
    const int numFrames = 1 + (sfinfo.frames - NFFT) / HOP;
    const double sr     = sfinfo.samplerate;
    double origFps      = sr / HOP;
    int outFps          = static_cast<int>(origFps / 6 + 0.5); // decimate by 6 for ~31fps

    //VideoWriter setup: choose your canvas size (e.g. 800x600)
    cv::Size canvasSize(800, 600);
    cv::VideoWriter writer(
        "visualizer.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        outFps,
        canvasSize,
        true
    );
    if (!writer.isOpened()) {
        std::cerr << "Error: could not open output video\n";
        return 1;
    }

    // Prepare FFTW
    std::vector<double> in(NFFT);
    std::vector<fftw_complex> out(freqBins);
    fftw_plan plan = fftw_plan_dft_r2c_1d(NFFT, in.data(), out.data(), FFTW_MEASURE);

    // Build Hann window
    std::vector<double> window(NFFT);
    for (int i = 0; i < NFFT; ++i)
        window[i] = 0.5 * (1 - cos(2 * M_PI * i / (NFFT - 1)));

    // Main loop: STFT + shape drawing
    for (int t = 0; t < numFrames; ++t) {
        int offset = t * HOP;
        // mix to mono and window
        for (int i = 0; i < NFFT; ++i) {
            double sum = 0;
            for (int c = 0; c < chan; ++c)
                sum += audio[(offset + i) * chan + c];
            in[i] = sum / chan * window[i];
        }
        fftw_execute(plan);

        // compute magnitudes for bands
        int NBANDS = 8;
        int binsPerBand = freqBins / NBANDS;
        std::vector<float> bands(NBANDS);
        for (int b = 0; b < NBANDS; ++b) {
            double acc = 0;
            for (int k = b * binsPerBand; k < (b+1) * binsPerBand && k < freqBins; ++k)
                acc += std::hypot(out[k][0], out[k][1]);
            bands[b] = static_cast<float>(acc / binsPerBand);
        }

        // create blank frame
        cv::Mat frame(canvasSize, CV_8UC3, cv::Scalar(10, 10, 30)); // dark background

        // draw circles
        cv::Point center(canvasSize.width/2, canvasSize.height/2);
        float maxRadius = std::min(canvasSize.width, canvasSize.height) * 0.45f;
        for (int b = 0; b < NBANDS; ++b) {
            float normVal = bands[b] / (*std::max_element(bands.begin(), bands.end()) + 1e-6f);
            float radius = normVal * maxRadius;
            if (radius < 60){radius = radius + 60;}
            int thickness = 2;
            int colorHue = static_cast<int>(b * 180 / NBANDS);
            cv::Scalar color;
            cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(colorHue, 200, 255));
            cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
            color = cv::Scalar(hsv.data[0], hsv.data[1], hsv.data[2]);
            cv::circle(frame, center, static_cast<int>(radius), color, thickness);
        }

       
        // write frame
        if (t % 6 == 0)
            writer.write(frame);
    }

    // cleanup
    fftw_destroy_plan(plan);
    writer.release();

    //Mux audio + video
    std::filesystem::path inPath(argv[1]);
    std::string outName = inPath.stem().string() + "_viz.mp4";
    std::string cmd =
        "ffmpeg -y -i visualizer.mp4 -i \"" + std::string(argv[1]) + "\" "
        "-map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest \"" + outName + "\"";
    std::cout << "Muxing to " << outName << "...\n";
    int ret = std::system(cmd.c_str());
    if (ret != 0) std::cerr << "ffmpeg mux failed code " << ret << "\n";
    else std::cout << "Done!\n";

    return 0;
}
