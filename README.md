# CNN Implementation on PYNQ: A Project Report

## Abstract

This report details the implementation of a Convolutional Neural Network (CNN) on the PYNQ (Python Productivity for Zynq) platform. The project leverages the hardware acceleration capabilities of the Zynq SoC (System on Chip) to perform image classification. The provided code utilizes the PYNQ libraries to interact with a pre-compiled hardware overlay (`cnn.bit`) that contains the CNN IP (Intellectual Property) core. This report will cover the project's introduction, methodology, expected results based on the implemented CNN architecture, and a final conclusion on the work done. A proposed system based on this implementation is also detailed. The primary objective of this project is to demonstrate the efficient utilization of the PYNQ framework for accelerating CNN-based image classification tasks.

## Introduction

Convolutional Neural Networks (CNNs) have emerged as a dominant technique in various computer vision applications, including image classification, object detection, and image segmentation. However, CNNs are computationally intensive, making their real-time deployment challenging on resource-constrained platforms. FPGAs (Field-Programmable Gate Arrays) offer a compelling solution by providing hardware acceleration for CNNs, leading to significant performance improvements compared to traditional CPU-based implementations.

PYNQ simplifies the development process by enabling designers to use Python to program Zynq SoCs, combining the flexibility of software programming with the performance of hardware acceleration.  This project showcases a PYNQ-based implementation of a CNN for image classification, using a pre-designed hardware overlay. The goal is to accelerate the CNN inference process by offloading the computationally intensive tasks to the FPGA fabric.

## Methodology

The methodology involves the following steps:

1.  **Hardware Overlay Loading:** The `pynq.Overlay` class is used to load the `cnn.bit` file, which contains the hardware description of the CNN IP core and the associated DMA (Direct Memory Access) controllers. This step configures the FPGA with the hardware implementation of the CNN.

    ```python
    from pynq import Overlay
    overlay = Overlay("./cnn.bit")
    ```

2.  **DMA Controller Instantiation:** The DMA controllers are instantiated to facilitate data transfer between the host system (CPU) and the FPGA fabric. Two DMA controllers are used: `dma0` for sending the input image to the CNN IP and `dma1` for receiving the output probabilities from the CNN IP.

    ```python
    dma0 = overlay.dma0     # DMA for sending input image
    dma1 = overlay.dma1     # DMA for receiving output probabilities
    ```

3.  **Memory Allocation:** PYNQ's `allocate` function is used to allocate contiguous memory buffers in the host system's RAM for storing the input image and the output probabilities.  The `image_in` buffer is allocated to hold the flattened 30x30x3 image, and the `probability_out` buffer is allocated to store the classification probabilities for 43 classes. A `class_out` buffer is also allocated (though not used in the provided code).

    ```python
    from pynq import allocate
    import numpy as np

    image_in = allocate(shape=(30*30*3,), dtype=np.float32)  # Your image input
    probability_out = allocate(shape=(43,), dtype=np.float32)  # 43 classes
    class_out = allocate(shape=(1,), dtype=np.float32)  # Final class output
    ```

4.  **Image Preprocessing:** A sample 30x30x3 RGB image is defined as a NumPy array. This image is then preprocessed by removing the singleton dimension and flattening it into a 1D array to be compatible with the DMA transfer.

    ```python
    custom_image = np.array([
        # ... (image data) ...
    ], dtype=np.float32)

    custom_image = np.squeeze(custom_image, axis=2)  # Now (30, 30, 3)
    custom_image_flattened = custom_image.reshape(-1)
    image_in[:] = custom_image_flattened
    ```

5.  **CNN IP Core Control:** The CNN IP core is accessed using the `overlay.cnn` object.  A reset sequence is performed on the CNN IP core by writing to its control registers. This ensures the IP is in a known state before starting the inference.

    ```python
    cnn_ip = overlay.cnn
    import time
    cnn_ip.write(0x00, 0x04)  # Reset
    time.sleep(0.01)           # Small delay
    cnn_ip.write(0x00, 0x00)  # Clear reset
    ```

6.  **Data Transfer and CNN Execution:** The `dma0.sendchannel.transfer()` function is used to initiate the DMA transfer of the preprocessed image data from the `image_in` buffer to the FPGA fabric.  Simultaneously, the `dma1.recvchannel.transfer()` function is used to prepare the DMA to receive the output probabilities from the FPGA fabric into the `probability_out` buffer. The CNN IP core is then started by setting the `ap_start` bit in its control register. The `dma0.sendchannel.wait()` and `dma1.recvchannel.wait()` functions are used to block until the DMA transfers are complete.

    ```python
    dma0.sendchannel.transfer(image_in)
    dma1.recvchannel.transfer(probability_out)
    cnn_ip.write(0x00, 0x01)  # Set ap_start=1
    dma0.sendchannel.wait()
    dma1.recvchannel.wait()
    ```

7.  **Result Interpretation:** After the DMA transfer is complete, the output probabilities stored in the `probability_out` buffer are analyzed to determine the predicted class.  The `np.argmax()` function is used to find the index of the class with the highest probability.  The predicted class and the probabilities for all classes are then printed to the console.

    ```python
    predicted_class = np.argmax(probability_out)
    print(f"Predicted Class: {predicted_class}")

    print("\nClass Probabilities:")
    for i, prob in enumerate(probability_out):
        print(f"Class {i}: {prob:.4f}")
    ```

## Expected Results

Based on the code, the CNN is configured to classify images into one of 43 classes.  The `probability_out` buffer will contain a set of 43 floating-point numbers representing the probabilities for each class. The expected output is the class with the highest probability.

The provided code sample is expected to output:

*   The predicted class (an integer between 0 and 42).
*   The probability for each of the 43 classes.

In the given output, the predicted class is 15 and the probability for class 15 is 1.0000.  This suggests that the CNN is highly confident in its classification of the input image. The CNN architecture details and the training dataset influence the prediction accuracy.

## Conclusion

This project demonstrates the successful implementation of a CNN on the PYNQ platform. The use of DMA controllers enables efficient data transfer between the host system and the FPGA fabric, allowing for hardware acceleration of the CNN inference process.  The PYNQ framework simplifies the development process by providing a high-level Python interface for interacting with the hardware.

The results show the CNN can classify a given input image. Further work can be done to validate these results against a broader dataset of images to further analyze the CNN architecture and its suitability for the given classification task.

## Proposed System

Based on this implementation, a more complete system could be built:

1.  **Real-time Image Acquisition:** Integrate a camera interface to capture images in real-time.  This would require developing a hardware interface to acquire image data from the camera and transfer it to the FPGA.

2.  **Dynamic Overlays:** Implement the ability to load different CNN models (hardware overlays) dynamically. This would allow the system to be reconfigured for different image classification tasks without requiring a complete system reboot.

3.  **Web Interface:** Develop a web-based interface for controlling the system and visualizing the results.  This would allow users to interact with the system remotely and view the classification results in real-time.

4.  **Training on the edge:** Explore training the CNN directly on the PYNQ platform, potentially using techniques like transfer learning to adapt pre-trained models to specific applications.  This would require significant modifications to the hardware overlay and the Python code.
