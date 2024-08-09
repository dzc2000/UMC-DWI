# UMVC
MICCAI 2024: Uncertainty-Aware Multi-View Learning for Prostate Cancer Grading with DWI


## Dataset

To prepare your DWI dataset, you can use the following code to process the data and extract the middle four slices from each `.nii` file:

```angular2html
import nibabel as nib
import os
from PIL import Image

def save_four_slices(nii_folder_path, output_folder_path):
    # Create output directory
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Iterate through all .nii files and save the middle four slices
    for nii_file_name in os.listdir(nii_folder_path):
        nii_file_path = os.path.join(nii_folder_path, nii_file_name)
        # Load the .nii file
        img = nib.load(nii_file_path)

        # Get image data
        img_data = img.get_fdata()

        # Get the indices of the middle four slices
        middle_slice_index = img_data.shape[-1] // 2
        if middle_slice_index - 2 >= 0 and middle_slice_index + 2 <= img_data.shape[2]:
            name = nii_file_name.split('.')[0]
            print(name)
            # Generate file names
            count = 0
            for i in range(middle_slice_index - 2, middle_slice_index + 2):  # Save the middle four slices
                output_file_path = os.path.join(output_folder_path, f'{name}_slice{count}.png')
                
                count += 1
                # Save the slice as an image file without displaying it
                img_slice = img_data[:, :, i]
                img_slice = (img_slice).astype('uint8')  # Convert to 8-bit integer
                img_pil = Image.fromarray(img_slice)

                # Save the image file
                img_pil.save(output_file_path)
```

## Train
To train the models, go to the corresponding directory, and run the command

```python train.py```


