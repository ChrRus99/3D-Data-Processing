Prerequisites (in debian-based distro):

sudo apt install build-essential cmake libopencv-dev libeigen3-dev

Build and run the executable:

mkdir build
cd build
cmake ..
make
cd ../bin/
./opencv_examples

The code is fully commented, and the used OpenCV functions are documented at:

https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html 
