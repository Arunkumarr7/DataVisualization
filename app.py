import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.robust.scale import mad
import numpy as np

# --- Page Setup ---
st.set_page_config(layout="wide") # Use wide layout for better visuals
st.title("📊 Dynamic Dataset Analyzer")
st.markdown("Upload a CSV file to begin the exploratory data analysis (EDA).")
st.markdown("---")

# --- 1. File Uploader (The Core Interaction) ---
uploaded_file = st.file_uploader("Choose a CSV file (e.g., student_data.csv, iris.csv)", type="csv")

# Only proceed if a file has been uploaded
if uploaded_file is not None:
    try:
        # Read the file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Identify numerical columns for analysis
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # --- 2. Dynamic Analysis Navigation ---
        st.header("1. Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("A. Raw Data (First 5 Rows)")
            st.dataframe(df.head())
        
        with col2:
            st.subheader("B. Data Consistency Check")
            st.write("**Missing Values per Column:**")
            st.dataframe(df.isnull().sum(), use_container_width=True)
            if df.isnull().sum().sum() == 0:
                st.success("The dataset is clean (No missing values).")
            else:
                st.warning("Warning: Missing values detected!")

        st.markdown("---")

        # --- 3. Statistical Summary ---
        st.header("2. Descriptive Statistics")
        st.subheader("C. Central Tendency & Spread")
        st.dataframe(df.describe().T)
        
        # --- 4. Visual Analysis Selector ---
        st.header("3. Visual Analysis")
        st.markdown("Select a numerical column for distribution plots and relationship analysis.")
        
        if not numerical_cols:
            st.warning("No numerical columns found for visual analysis.")
        else:
            # Dropdown to select the primary column for distribution plots
            selected_column = st.selectbox("Select a primary column:", numerical_cols)
            
            # --- Box Plot & Histogram ---
            st.subheader(f"D/G. Distribution of: {selected_column}")
            
            col3, col4 = st.columns(2)
            
            # Box Plot
            with col3:
                fig_box, ax_box = plt.subplots(figsize=(4, 3))
                sns.boxplot(y=df[selected_column], ax=ax_box)
                ax_box.set_title(f"Box Plot: {selected_column}")
                st.pyplot(fig_box)
            
            # Histogram
            
            
            # --- Correlation Heatmap (H) ---
            st.subheader("H. Correlation Matrix")
            correlation_matrix = df[numerical_cols].corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(8, 7))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            ax_corr.set_title("Correlation Between Numerical Variables")
            st.pyplot(fig_corr)

            st.markdown("---")
            #st.header("4. Conclusion")
            #st.success("Analysis Complete: Review the statistics and visualizations above.")

    except Exception as e:
        st.error(f"Error processing file: Ensure the file is a valid CSV and not corrupted. Details: {e}")

else:
    # Message displayed if no file is uploaded
    st.info("Please upload a CSV file to populate the analysis sections.")