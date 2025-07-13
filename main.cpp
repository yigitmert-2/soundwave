#include <iostream>
#include <ostream>
#include <sndfile.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

int main(int argc, char** argv){
    if(argc<2){
        fprintf(stderr,"Usage: %s input.wav\n", argv[0]);
        return 1;
    }
    //–– 1) Read audio into a float buffer ––
    SF_INFO sfinfo;
    SNDFILE* snd = sf_open(argv[1], SFM_READ, &sfinfo);
    if(!snd){ perror("sf_open"); return 1; }
    std::vector<float> audio(sfinfo.frames * sfinfo.channels);
    sf_readf_float(snd, audio.data(), sfinfo.frames);
    
    sf_close(snd);

    //–– 2) STFT params ––
    const int    chan       = sfinfo.channels;
    const int    NFFT       = 1024;   // window size
    const int    HOP        = NFFT/4; // 75% overlap
    const int    numFrames  = 1 + (sfinfo.frames - NFFT)/HOP;
    const double sr         = sfinfo.samplerate;
    double       fps        = sfinfo.samplerate / HOP ; 

    //video writer
    cv::VideoWriter writer(
        "spectogram.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        cv::Size(NFFT/2+1, NFFT/2+1),
        true
    );


    //–– 3) Prepare FFTW ––
    std::vector<double>   in(NFFT);
    std::vector<fftw_complex> out(NFFT/2+1);
    fftw_plan            plan = fftw_plan_dft_r2c_1d(NFFT, in.data(), out.data(), FFTW_MEASURE);

    //–– 4) Prepare OpenCV image ––
    //    rows = freq bins, cols = time bins
    cv::Mat spec = cv::Mat::zeros(NFFT/2+1, numFrames, CV_32F);

    //–– 5) Window function (Hann) ––
    std::vector<double> window(NFFT);
    for(int i=0;i<NFFT;i++)
        window[i] = 0.5 * (1 - cos(2*M_PI*i/(NFFT-1)));

    //–– 6) Slide and FFT ––
    for(int t=0; t<numFrames; t++){
        int offset = t*HOP;
        // mix down if stereo > mono
        for(int i=0; i<NFFT; i++){
            double sum = 0;
            for(int c=0; c<chan; c++)
                sum += audio[(offset+i)*chan + c];
            in[i] = (sum/chan) * window[i];
        }
        fftw_execute(plan);
        for(int k=0; k<=NFFT/2; k++){
            double mag = sqrt(out[k][0]*out[k][0] + out[k][1]*out[k][1]);
            spec.at<float>(k,t) = (float)mag;
        }
    }
    fftw_destroy_plan(plan);

    //–– 7) Log-scale & normalize 0–255 ––
    cv::Mat logSpec;
    spec += 1e-6;                        // avoid log(0)
    cv::log(spec, logSpec);             // natural log
    cv::normalize(logSpec, logSpec, 0, 255, cv::NORM_MINMAX);
    logSpec.convertTo(logSpec, CV_8U);

    //–– 8) (Optional) apply a color map ––
    cv::Mat colorSpec;
    cv::applyColorMap(logSpec, colorSpec, cv::COLORMAP_INFERNO);

    //–– 9) Save it ––
    cv::imwrite("spectrogram.png", colorSpec);
    printf("Saved spectrogram.png (%d×%d)\n", colorSpec.cols, colorSpec.rows);

    return 0;
}
