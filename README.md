# PrivCom – Privacy-Preserving Compression

## Topic

**Privacy-Preserving Image Compression** using matrix factorization and region-of-interest masking.

## Team Members

* Samantha W (23PW25)
* Sruthi K (23PW33)

## Overview

PrivCom is a Python package that compresses images while protecting sensitive information. It implements two approaches:

1. **PMC (Privacy-preserving Matrix Compression):** Compresses images using SVD and adds controlled noise to embed privacy.
2. **ROI-based Compression:** Detects sensitive regions (like faces) and applies masking before compressing the image.

Both approaches balance storage efficiency and privacy, making them suitable for healthcare, surveillance, and cloud applications.

## Features

* Compress images while preserving privacy.
* PMC: SVD-based global privacy compression.
* ROI: Targeted masking of sensitive regions before JPEG compression.
* Computes metrics: PSNR, MSE, and compression ratio.
* Generates comparison plots of original, PMC, and ROI-processed images.
* Supports batch processing of multiple images.

## Conclusion

PrivCom demonstrates practical methods to achieve both privacy and compression. PMC offers global protection through mathematical transformations, while ROI masking secures sensitive areas selectively. Together, these methods show how privacy-preserving compression can be applied in real-world scenarios.
