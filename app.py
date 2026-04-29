import streamlit as st
import pandas as pd
import glob
import os
from polygon_congruence import parse_polygon_text, analyze, test_congruence, plot_pair

# Set page config for a wider layout
st.set_page_config(page_title="Polygon Congruence Tool", layout="wide")

st.title("Polygon Congruence Tool")
st.markdown("Upload `.txt` files containing 3D points to test for congruence.")

# Initialize session state so data persists when buttons are clicked
if 'polygons' not in st.session_state:
    st.session_state.polygons = {}

# --- Sidebar Controls ---
with st.sidebar:
    # 1. Upload Data
    st.header("1. Data Management")
    uploaded_files = st.file_uploader("Upload .txt files", accept_multiple_files=True, type=['txt'])

    # Load Defaults Button
    if st.button("Load Example Polygons"):
        example_files = glob.glob("sample_data/*.txt")
        for fpath in example_files:
            name = os.path.basename(fpath)
            with open(fpath, 'r') as f:
                content = f.read()
                if name not in st.session_state.polygons:
                    try:
                        pts = parse_polygon_text(content, name)
                        st.session_state.polygons[name] = analyze(pts, name)
                    except Exception as e:
                        st.error(f"Error loading {name}: {e}")
        st.rerun()

    # UNLOAD/CLEAR Button
    if st.button("Clear All Polygons", type="secondary"):
        st.session_state.polygons = {}
        st.rerun()
    
    st.divider()
    allow_si = st.checkbox('Allow self-intersecting', value=True)
    
    # Process Uploaded Files
    if uploaded_files:
        for file in uploaded_files:
            name = file.name
            content = file.getvalue().decode('utf-8')
            if name not in st.session_state.polygons:
                try:
                    pts = parse_polygon_text(content, name)
                    st.session_state.polygons[name] = analyze(pts, name)
                except Exception as e:
                    st.error(f"Error loading {name}: {e}")

# --- Main Content Area ---
if st.session_state.polygons:
    # 1. Catalog Table
    st.header("Loaded Polygons")
    catalog = [{"Name": p.name, "Vertices": len(p.pts3d), "Area": round(p.area, 4), "Valid": p.valid} 
               for p in st.session_state.polygons.values()]
    st.dataframe(pd.DataFrame(catalog), use_container_width=True)

    # 2. Batch Analysis (Find All)
    st.header("Batch Congruence Check")
    if st.button("Run Global Comparison"):
        names = [k for k, v in st.session_state.polygons.items() if v.valid]
        found = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                res = test_congruence(st.session_state.polygons[names[i]], 
                                     st.session_state.polygons[names[j]], 
                                     allow_self_intersecting=allow_si)
                if res['congruent']:
                    found.append({"Polygon A": names[i], "Polygon B": names[j], "Max Error": f"{res['max_err']:.2e}"})
        
        if found:
            st.success(f"Found {len(found)} congruent pairs!")
            st.table(pd.DataFrame(found))
        else:
            st.info("No congruent pairs detected.")

    # 3. Individual Pair Test 
    st.divider()
    st.header("Manual Pair Inspection")
    
    valid_names = [name for name, p in st.session_state.polygons.items() if p.valid]
    
    if len(valid_names) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            sel_a = st.selectbox("Select Polygon A", valid_names, index=0)
        with col2:
            sel_b = st.selectbox("Select Polygon B", valid_names, index=1)

        if st.button("Test Congruence", type="primary"):
            if sel_a == sel_b:
                st.warning("Please select two different polygons.")
            else:
                poly_a = st.session_state.polygons[sel_a]
                poly_b = st.session_state.polygons[sel_b]
                
                result = test_congruence(poly_a, poly_b, allow_self_intersecting=allow_si)
                
                if result['congruent']:
                    st.success("✅ Congruent!")
                    st.write(f"**Shift:** {result['shift']} | **Reversed:** {result['reversed_order']} | **Reflection:** {result['used_reflection']}")
                else:
                    st.error(f"❌ Not Congruent: {result['reason']}")
                
                # Display the plot
                fig = plot_pair(poly_a, poly_b, result)
                st.pyplot(fig)
    else:
        st.info("Load at least two valid polygons to perform manual inspection.")
else:
    st.info("Please upload polygon files or load examples in the sidebar to begin.")