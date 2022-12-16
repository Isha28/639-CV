# 639-CV

The problem we are trying to solve is providing access to digital images for the visually impaired. In 2022, 2.2 billion people are reported to have a near or distance vision impairment. This means that many of them have limited access to the digital world as most of it is reliant on sight. One way that visually impaired people navigate the internet is through text to speech applications. However, this method does have its limitations because these simple text to speech solutions cannot read the text from images. Thus, we propose Image to Speech Translation solution which takes an input image and extracts text from it. The extracted text is translated to speech.

This problem is important because it allows visually challenged people to get a deeper enjoyment and utilization of the internet and perhaps help them navigate the outside world. Ultimately, this all allows them to be more autonomous in the real world and online.

The solution to our proposed problem requires three main steps - Image Pre-processing, Image to Text with OCR, Text to Speech synthesis.

Step 1: Image Pre-processing : 
Below are the important pre-processing techniques we applied on the input image to enhance the quality of the image and extract the right text. We are using the OpenCV python package for this purpose. This step will greatly improve user experience and eliminate some of the current limitations of the OCR systems.

Grayscale Conversion : We are using OpenCV cvtColor function to convert the colored image to grayscale image. <br>
Threshold and Binarization : This step converts the grayscale image into a binary image with only black and white color. This is done so that Tesseract OCR can identify text easily. <br>
Noise Removal : We used openCV erode and dilate method to remove small white noises and increase the object area of the input image. We applied median blur filtering technique to preserve the edges of image while removing noise. <br>

Step 2: Image to Text : 
We used open source Tesseract Optical Character Recognition engine for obtaining text from images [6]. This library is proven to provide better quality output. We provided the required configuration settings in a Python program to read through categories such as texts, numbers and special characters from images and output them with acceptable level of accuracy. This needed trial and error effort to find the right configuration for each of the categories. We have tested the code with rich set of images including images with short words, magazines, and handwritten notes to validate the correctness of the module.

Step 3: Text to Speech :
In this final step, the resulting text is sent to the text-to-speech solution. We used an open source TTS solution by Coqui, which is derived from the Mozilla TTS solution. This TTS solution is made up of three components, the TTS model itself, the dataset used, and the vocoder model. We used the pretrained TTS model known as Tacotron2 with Dynamic Convolutional Attention (Tacotron2 DCA). We then used the LJSpeech dataset, which is a public domain english speech dataset and for the vocoder model we chose the Multi-Band MelGAN because it was the fastest model.
