import streamlit as st
import numpy as np
from PIL import Image
import io

# Only import packages that are definitely available
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Page configuration
st.set_page_config(
    page_title="Neuroflux - Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Neuroflux</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Brain Tumor Segmentation & Analysis")
    
    # Show package status
    with st.expander("Package Status"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if HAS_CV2:
                st.success("OpenCV (cv2) - Ready")
            else:
                st.error("OpenCV (cv2) - Missing")
        with col2:
            if HAS_PLOTLY:
                st.success("Plotly - Ready")  
            else:
                st.error("Plotly - Missing")
        with col3:
            st.info("ℹNeuroflux - Optional")
    
    # Sidebar
    with st.sidebar:
        st.header("About Neuroflux")
        st.write("""
        **Features:**
        • Upload MRI and CT scans
        • Tumor segmentation  
        • Grad-CAM Heatmap visualization
        """)
        
        st.header("Performance")
        st.metric("Accuracy", "99%")
        st.metric("Precision", "99.2%") 
        st.metric("Dice Score", "53.3%")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload & Configure")
        
        # Scan type
        scan_type = st.selectbox("Scan Type:", ["MRI", "CT"])
        
        # File uploader
        uploaded_file = st.file_uploader(
            f"Upload {scan_type} scan:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a brain scan image"
        )
        
        # Settings
        with st.expander("Settings"):
            confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
            show_overlay = st.checkbox("Show Segmentation Overlay", True)
            overlay_opacity = st.slider("Overlay Opacity", 0.1, 1.0, 0.6)
    
    with col2:
        st.header("📈 Analysis Results")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.subheader("Original Scan")
            st.image(image, caption=f"Uploaded {scan_type} scan", use_column_width=True)
            
            # Analyze button
            if st.button("Analyze Scan", type="primary"):
                analyze_image(image, scan_type, confidence, show_overlay, overlay_opacity)
        else:
            st.info("Upload a brain scan to get started!")
            show_demo()

def analyze_image(image, scan_type, confidence, show_overlay, overlay_opacity):
    """Analyze the uploaded image"""
    
    with st.spinner(f"Analyzing {scan_type} scan..."):
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        h, w = img_array.shape[:2]
        
        # Create demo results
        results = create_demo_segmentation(img_array)
        
        # Show results in tabs
        tab1, tab2, tab3 = st.tabs(["Segmentation", "Analysis", "Report"])
        
        with tab1:
            st.subheader("GBM Segmentation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original", use_column_width=True)
            
            with col2:
                if show_overlay and HAS_CV2:
                    # Create colored overlay
scm-history-item:/workspaces/neuroflux?%7B%22repositoryId%22%3A%22scm0%22%2C%22historyItemId%22%3A%22d8d1a314fcd562502422c237b7b053fa75fd8483%22%2C%22historyItemParentId%22%3A%224fafd1fb578373ac64cc0b69abd1364f8dbf2019%22%2C%22historyItemDisplayId%22%3A%22d8d1a31%22%7D                    overlay = create_overlay(img_array, results['mask'], overlay_opacity)
                    st.image(overlay, caption="With Segmentation", use_column_width=True)
                else:
                    st.image(results['mask'], caption="Segmentation Mask", use_column_width=True)
        
        with tab2:
            st.subheader("Analysis Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{results['confidence']:.1%}")
            with col2:
                st.metric("Tumor Volume", f"{results['volume']:.1f} mm³")
            with col3:
                st.metric("Affected Area", f"{results['area_ratio']:.1%}")
            
            # Show heatmap if plotly available
            if HAS_PLOTLY:
                st.subheader("Attention Heatmap")
                fig = px.imshow(
                    results['heatmap'],
                    color_continuous_scale='hot',
                    title="AI Attention Map"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Medical Report")
            generate_simple_report(results, scan_type)

def create_demo_segmentation(img_array):
    """Create demo segmentation results"""
    h, w = img_array.shape[:2]
    
    # Create synthetic tumor mask
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    
    # Add some "tumor" regions
    if HAS_CV2:
        cv2.circle(mask, (center_x + 20, center_y - 15), 30, 255, -1)
        cv2.circle(mask, (center_x - 25, center_y + 10), 20, 128, -1)
    else:
        # Fallback without cv2
        y, x = np.ogrid[:h, :w]
        mask1 = ((x - (center_x + 20))**2 + (y - (center_y - 15))**2) <= 30**2
        mask2 = ((x - (center_x - 25))**2 + (y - (center_y + 10))**2) <= 20**2
        mask[mask1] = 255
        mask[mask2] = 128
    
    # Create attention heatmap
    heatmap = np.random.rand(h, w) * 0.3
    heatmap[mask > 0] += 0.7
    
    # Calculate metrics
    tumor_pixels = np.sum(mask > 0)
    total_pixels = h * w
    area_ratio = (tumor_pixels / total_pixels) * 100
    
    return {
        'mask': mask,
        'heatmap': heatmap,
        'confidence': 0.87,
        'volume': 1250.5,
        'area_ratio': area_ratio
    }

def create_overlay(image, mask, opacity):
    """Create colored overlay on image"""
    if not HAS_CV2:
        return mask
    
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    
    # Color the mask regions
    colored_mask[mask == 255] = [255, 0, 0]  # Red for tumor
    colored_mask[mask == 128] = [255, 255, 0]  # Yellow for edema
    
    # Blend with original
    result = cv2.addWeighted(image, 1-opacity, colored_mask, opacity, 0)
    return result

def generate_simple_report(results, scan_type):
    """Generate a simple medical report"""
    st.markdown(f"""
    ## Analysis Report
    
    **Scan Type:** {scan_type}  
    **Analysis Date:** Today  
    **AI Model:** Neuroflux v1 Demo
    
    ### Findings
    - **Tumor Detection:** Positive
    - **Confidence Level:** {results['confidence']:.1%}
    - **Estimated Volume:** {results['volume']:.1f} mm³
    - **Affected Brain Area:** {results['area_ratio']:.1f}%
    
    ### Disclaimer
    This is a demonstration analysis. Real medical diagnosis would require subsequent review from a pathologist and potentially additional imaging if required
    
    **Not for clinical use.**
    """)

def show_demo():
    """Show demo when no image uploaded"""
    st.write("**What you'll see after uploading:**")
    
    # Create sample visualization
    demo_data = np.random.rand(50, 50)
    demo_data[15:35, 20:30] += 0.8  # "tumor" region
    
    if HAS_PLOTLY:
        fig = px.imshow(
            demo_data,
            color_continuous_scale='hot',
            title="Sample: AI Attention Map"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.image(demo_data, caption="Sample Analysis", use_column_width=True)

if __name__ == "__main__":
    main()