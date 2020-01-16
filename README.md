Azure Cognitive Services - Custom Vision Sample Program for OpenCV DNN Module
=============================================================================

This repository is sample program of inference using OpenCV DNN module with Azure Custom Vision model written in native C++.  
This sample program inference on local using model that trained on cloud.  


Environment
-----------
* Visual Studio 2017/2019 / GCC 4.9 (or later)  
* OpenCV 4.2.0 (or later)  
* CMake 3.7.2 (latest release is preferred)  

Custom Vision
-------------
Please train your model using Custom Vision of Azure Cognitive Services.  
After trained, You can export your trained model (ONNX format).  
Please unzip, and copy these files (<code>model.onnx</code>, <code>labels.txt</code>) to source code directory.

License
-------
Copyright &copy; 2020 Tsukasa SUGIURA  
Distributed under the [MIT License](http://www.opensource.org/licenses/mit-license.php "MIT License | Open Source Initiative").

Contact
-------
* Tsukasa Sugiura  
    * <t.sugiura0204@gmail.com>  
    * <http://unanancyowen.com>  

Reference
---------
* Azure Cognitive Services - Custom Vision | Microsoft  
  <https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/>  
  <https://www.customvision.ai/>

* OpenCV API Reference | OpenCV  
  <https://docs.opencv.org/master/>

* UnaNancyOwen/OpenCVDNNSample | GitHub  
  <https://github.com/UnaNancyOwen/OpenCVDNNSample>
