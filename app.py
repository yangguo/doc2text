"""
Doc2Text - Streamlit application for document text and table extraction

This application provides a user interface for extracting text and tables from various
document formats (PDF, DOCX, DOC, WPS, images) using GPT-4o's vision capabilities.

The app has three main functionalities:
1. Document Upload: Upload various document formats for processing
2. Text Extraction: Extract and download text from documents
3. Table Recognition: Extract tables from documents and save as CSV files

Author: Original author
License: Apache-2.0
"""

import datetime
import os
import pandas as pd
import streamlit as st
from doc2text import (
    convert_df2zip,
    convert_table2zip,
    docxconvertion,
    extract_table,
    extract_text,
    get_uploadfiles,
    remove_uploadfiles,
    save_uploadedfile,
)

# Define the path for uploaded files
uploadpath = "uploads/"

def main():
    """
    Main function to run the Streamlit application.
    Handles the UI and logic for document processing.
    """
    st.title("Doc2Text")
    st.subheader("Document Text and Table Extraction with GPT-4o")
    
    # Define the available menu options in the sidebar
    menuls = ["Document Upload", "Text Extraction", "Table Recognition"]
    menu = st.sidebar.selectbox("Select Action", menuls)
    
    if menu == "Document Upload":
        handle_document_upload()
    elif menu == "Text Extraction":
        handle_text_extraction()
    elif menu == "Table Recognition":
        handle_table_recognition()

def handle_document_upload():
    """Handle the document upload functionality."""
    st.header("Upload Documents")
    st.write("Upload your documents for processing. Supported formats: PDF, DOCX, DOC, WPS, and various image formats.")
    
    # File uploader widget
    uploaded_file_ls = st.file_uploader(
        "Select documents to upload",
        type=["docx", "pdf", "doc", "wps", "bmp", "png", "jpg", "jpeg", "tiff"],
        accept_multiple_files=True,
        help="You can upload multiple files at once",
    )
    
    # Save each uploaded file
    for uploaded_file in uploaded_file_ls:
        if uploaded_file is not None:
            save_uploadedfile(uploaded_file, uploadpath)
    
    # Button for removing all files
    remove_button = st.sidebar.button("Delete All Documents")
    if remove_button:
        remove_uploadfiles(uploadpath)
        st.success("All documents have been deleted")
    
    # Automatic cleanup at midnight
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if current_time[-8:] == "00:00:00":
        remove_uploadfiles(uploadpath)
    
    # Display uploaded files
    display_uploaded_files()

def handle_text_extraction():
    """Handle the text extraction functionality."""
    st.header("Text Extraction")
    st.write("Extract text from your uploaded documents using GPT-4o's advanced OCR capabilities.")
    
    # Display uploaded files
    df = display_uploaded_files()
    if df is None:
        return
    
    # Format conversion button
    convert_button = st.sidebar.button("Convert Document Formats")
    if convert_button:
        with st.spinner("Converting document formats..."):
            docxconvertion(uploadpath)
        st.success("Format conversion complete!")
    
    # Text extraction button
    extract_button = st.sidebar.button("Extract Text")
    if extract_button:
        with st.spinner("Extracting text from documents..."):
            dfnew = extract_text(df, uploadpath)
        
        st.subheader("Extraction Results")
        st.table(dfnew)
        
        # Convert results to downloadable zip file
        with st.spinner("Preparing download..."):
            downloadname = convert_df2zip(dfnew, uploadpath)
        
        # Download button
        with open(downloadname, "rb") as f:
            st.download_button("Download Results", f, file_name="extracted_text.zip")

def handle_table_recognition():
    """Handle the table recognition functionality."""
    st.header("Table Recognition")
    st.write("Extract tables from documents and images using GPT-4o's vision capabilities.")
    
    # Display uploaded files
    df = display_uploaded_files()
    if df is None:
        return
    
    # Table extraction button
    convert_table = st.sidebar.button("Extract Tables")
    if convert_table:
        with st.spinner("Extracting tables from documents..."):
            tablels = extract_table(df, uploadpath)
        
        st.success(f"Successfully extracted {len(tablels)} tables!")
        
        # Prepare download
        with st.spinner("Preparing download..."):
            downloadname = convert_table2zip(tablels, uploadpath)
        
        # Download button
        with open(downloadname, "rb") as f:
            st.download_button("Download Tables", f, file_name="extracted_tables.zip")

def display_uploaded_files():
    """
    Display the list of uploaded files.
    
    Returns:
        pandas.DataFrame or None: DataFrame with file information or None if no files
    """
    # Get list of uploaded files
    filels = get_uploadfiles(uploadpath)
    
    if len(filels) > 0:
        # Convert file list to DataFrame and display
        df = pd.DataFrame({"File": filels})
        st.subheader("Uploaded Documents")
        st.write(df)
        return df
    else:
        st.error("No documents uploaded. Please upload documents first.")
        return None

if __name__ == "__main__":
    main()
