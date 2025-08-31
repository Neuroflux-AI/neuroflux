# Neuroflux

### Overview
Neuroflux is a multimodal Grad-CAM-enhanced U-Net-based deep learning model for Glioblastoma Multiforme segmentation

### Running the Neuroflux Library
Neuroflux is available as a public Python package as part of the Python Package Index! See instructions to use Neuroflux [here](https://pypi.org/project/neuroflux/). Predefined model weights for the MRI and CT scans are available on Neuroflux's Google Drive! Access mri_weights.h5 [here](https://drive.google.com/file/d/1-636ryo8Uz2M_km9HNxDxYOD1Dr464ly/view?usp=sharing) and access ct_weghts.pth [here](https://drive.google.com/file/d/1Ie2Q9MHubN3C4SqGm3XdHg1Keci1NjZF/view?usp=sharing). Download these files and upload them as instructed on our PyPI page. After uploading an MRI or CT scan, the model will generate segmented images and Grad-CAM heatmaps showing the areas of the brain where the model is focusing, showing areas of tumor damage.

### MRI Model Evaluation on the Test Set
1. Pixelwise Accuracy: 0.99
2. MeanIOU: 0.7067
3. Dice coefficient: 0.5332
4. Precision: 0.9922
5. Sensitivity: 0.9873
6. Specificity: 0.9973
7. Dice coef Necrotic: 0.445
8. Dice coef Edema: 0.6018
9. Dice coef Enhancing: 0.6564

### CT Model Evaluation on the Test Set
1. Classification Accuracy: 0.9769

### Contributions
Feel free to fork the repository and submit a pull request with any improvements. If you have any questions or run into issues, create an issue in the GitHub repository, and we will be happy to assist you!
