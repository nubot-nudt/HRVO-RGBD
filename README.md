# Hybrid Residuals based RGBD Visual Odometry
## Description
  In this work, we propose a novel hybrid residuals based RGBD visual odometry, where three kinds of complementary information are integrated in the joint optimization model. The reprojection residuals, the photometric residuals and the depth residuals are minimized together in the non-linear optimization process, and the robust cost function and the outliers ltering are employed in the iterative optimization to enhance the robustness of the iteration and maintain the optimality at the same time.
  
  This project cantians the basic implementation of HRVO-RGBD, in the files of se3optimization.h and se3optimization.cpp. The demonstration of applying this method in ORB-SLAM2 is also provided in this project.
  
  Maintainer: NuBot workshop, NUDT China - http://nubot.trustie.net and https://github.com/nubot-nudt
