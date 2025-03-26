# Neuroflux

### Overview
Neuroflux is a multimodal Grad-CAM-enhanced U-Net-based deep learning model for Glioblastoma Multiforme segmentation

### Running the Project in Google Colab
1. Navigate to neuroflux.ipynb and select **Open in Colab**
2. Select **File > Save a Copy in Drive**
3. Upload a Kaggle API key (Kaggle Account Settings > API > Create New Token) as a temporary file 
4. Run the notebook with **Runtime > Run all**

### Model Evaluation on the Test Set
1. Loss: 0.0319
2. Accuracy: 0.9901
3. MeanIOU: 0.6165
4. Dice coefficient: 0.5615
5. Precision: 0.9911
6. Sensitivity: 0.9885
7. Specificity: 0.997
8. Dice coef Necrotic: 0.4347
9. Dice coef Edema: 0.6882
10. Dice coef Enhancing: 0.6267  

### Additional Notes
1. **GPU Support:** Google Colab provides free GPU access, so using Colab will accelerate training. Ensure the Colab runtime is set to GPU by selecting Runtime > Change runtime type > Hardware accelerator > GPU.
2. **Model Output:** The notebook will generate segmented images and Grad-CAM heatmaps showing the areas of the brain where the model is focusing.

### Contributions
Feel free to fork the repository and submit a pull request with any improvements. If you have any questions or run into issues, create an issue in the GitHub repository, and we will be happy to assist you!
