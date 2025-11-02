import streamlit as st
import os
import tempfile
import neuroflux

st.title("Neuroflux AI")
st.write("Upload MRI or CT scans to view AI-generated tumor heat maps")

modality=st.radio("Select modality:", ["MRI", "CT"])

if modality == "MRI":
    input_flair = st.file_uploader("Upload FLAIR", type=["nii", "nii.gz"])
    input_t1ce = st.file_uploader("Upload T1ce", type=["nii", "nii.gz"])
    model_weights = os.path.join(os.path.dirname(__file__), "mri_.weights.h5")
    slice_num = st.number_input("Slice number", value=75, step=1)
elif modality == "CT":
    input_ct = st.file_uploader("Upload CT scan", type=["jpg", "jpeg", "png"])
    model_weights = os.path.join(os.path.dirname(__file__), "ct_weights.pth")

if st.button("Generate Heat map"):    
    if modality == "MRI" and input_flair and input_t1ce:
        with st.spinner("Generating heat map..."):
            
            with tempfile.TemporaryDirectory() as tmpdir:
                flair_path = os.path.join(tmpdir, input_flair.name)
                t1ce_path = os.path.join(tmpdir, input_t1ce.name)

                with open(flair_path, "wb") as f:
                    f.write(input_flair.read())
                with open(t1ce_path, "wb") as f:
                    f.write(input_t1ce.read())

                model = neuroflux.mri.prepare_mri_model(model_weights=model_weights, img_size=128)
                output_slice_path = neuroflux.mri.display_slice(folder=tmpdir, input_flair=input_flair.name, input_t1ce=input_t1ce.name, slice_num=slice_num, model=model)
                st.image(f"mri_gradcam_slice_{slice_num}.png")
                neuroflux.mri.display_grid(folder=tmpdir, input_flair=input_flair.name, input_t1ce=input_t1ce.name, model=model)
                st.image("mri_gradcam_grid.png")

    elif modality == "CT" and input_ct:
        with st.spinner("Generating heat map..."):

            with tempfile.TemporaryDirectory() as tmpdir:
                ct_path = os.path.join(tmpdir, input_ct.name)

                with open(ct_path, "wb") as f:
                    f.write(input_ct.read())

                model = neuroflux.ct.prepare_ct_model(model_weights=model_weights)
                neuroflux.ct.display_gradcam(folder=tmpdir, input=input_ct.name, model=model)
                st.image("ct_gradcam_grid.png")

    else:
        st.error("Please upload all required files first")
