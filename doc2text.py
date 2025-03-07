"""
Doc2Text - Intelligent Document Text and Table Extraction Library

This library provides functionality to extract text and tables from various document formats
using OpenAI's GPT-4o vision capabilities. It can process PDFs, Word documents (DOCX, DOC, WPS),
and various image formats to extract both text content and tabular data.

Key Features:
- Text extraction from documents using OCR and direct text extraction methods
- Table extraction from documents using GPT-4o's vision capabilities
- Support for multiple document formats including PDF, DOCX, DOC, WPS, PNG, JPG, TIFF, etc.
- Document format conversion using LibreOffice

This module is designed to be used with the accompanying Streamlit application (app.py)
but can also be used as a standalone library for document processing.

Author: Original author
License: Apache-2.0
"""

import fnmatch
import glob
import os
import re
import subprocess
import zipfile
import base64
from pathlib import Path
from dotenv import load_dotenv
import docx
import pandas as pd
import pdfplumber
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
import openai
import json

# Constants
Image.MAX_IMAGE_PIXELS = None  # Disable PIL image size limit

# Load environment variables (for OpenAI API key)
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    """
    Encode an image to base64 for sending to the OpenAI API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gpt4o_ocr_text(image_path):
    """
    Extract text from an image using GPT-4o's vision capabilities.
    
    This function sends the image to OpenAI's GPT-4o model and requests
    a complete text extraction with preserved layout.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Encode image to base64
        base64_image = encode_image(image_path)
        
        # Create message with the image
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an OCR assistant. Extract ALL text from the image exactly as it appears. Include line breaks and spacing. Only output the extracted text, nothing else."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract all text from this image, preserving the layout as much as possible."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in GPT-4o OCR: {str(e)}")
        return ""


def gpt4o_extract_table(image_path):
    """
    Extract table data from an image using GPT-4o's vision capabilities.
    
    This function sends the image to OpenAI's GPT-4o model and requests
    the extraction of tabular data, which is then converted to a pandas DataFrame.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        pandas.DataFrame: Extracted table as a DataFrame
    """
    try:
        # Encode image to base64
        base64_image = encode_image(image_path)
        
        # Create message with the image, specifically requesting table extraction
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a table extraction assistant. Extract the table data from the image and return it as a JSON array of arrays, where each inner array represents a row of the table. Preserve all data exactly as it appears in the table. Include headers if present. Return ONLY the JSON array."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the table data from this image as a JSON array of arrays. Each inner array should represent a row in the table."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=2048
        )
        
        # Extract the JSON from the response
        table_text = response.choices[0].message.content
        
        # Clean up the response to extract just the JSON part
        table_text = table_text.strip()
        if table_text.startswith("```json"):
            table_text = table_text.split("```json")[1]
        if table_text.startswith("```"):
            table_text = table_text.split("```")[1]
        if table_text.endswith("```"):
            table_text = table_text.rsplit("```", 1)[0]
        
        # Parse the JSON into a DataFrame
        table_data = json.loads(table_text)
        df = pd.DataFrame(table_data)
        
        return df
    except Exception as e:
        st.error(f"Error in GPT-4o table extraction: {str(e)}")
        return pd.DataFrame()


def docxurl2txt(url):
    """
    Extract text from a docx file.
    
    This function uses the python-docx library to extract text from all paragraphs
    in a Word document.
    
    Args:
        url (str): Path to the docx file
        
    Returns:
        str: Extracted text from the document with paragraphs separated by newlines
    """
    text = ""
    try:
        doc = docx.Document(url)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        text = "\n".join(fullText)
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
    return text


def pdfurl2txt(url):
    """
    Extract text from a PDF file using pdfplumber.
    
    This function attempts to extract text directly from a PDF document
    using the PDF's embedded text information.
    
    Args:
        url (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    result = ""
    try:
        with pdfplumber.open(url) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    result += txt
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    return result


def pdfurl2ocr(url, uploadpath):
    """
    Extract text from PDF using OCR by first converting pages to images.
    
    This function is used when direct text extraction from a PDF fails.
    It converts each page to an image and then applies OCR using GPT-4o.
    
    Args:
        url (str): Path to the PDF file
        uploadpath (str): Directory to temporarily store image files
        
    Returns:
        str: Extracted text using OCR
    """
    PDF_file = Path(url)
    image_file_list = []
    text = ""
    
    # Convert PDF pages to images
    pdf_pages = convert_from_path(PDF_file, 500)
    
    # Save each page as an image
    for page_enumeration, page in enumerate(pdf_pages, start=1):
        filename = os.path.join(uploadpath, f"page_{page_enumeration}.jpg")
        page.save(filename, "JPEG")
        image_file_list.append(filename)

    # Process each image with OCR
    for image_file in image_file_list:
        text += gpt4o_ocr_text(image_file)
        # Clean up temporary image file
        os.remove(image_file)

    return text


def docxurl2ocr(url, uploadpath):
    """
    Extract text from images embedded in a DOCX file using OCR.
    
    This function extracts all images from a DOCX file and processes
    them with OCR using GPT-4o to capture text that might be embedded
    in images within the document.
    
    Args:
        url (str): Path to the DOCX file
        uploadpath (str): Directory to temporarily store extracted images
        
    Returns:
        str: Extracted text from images found in the DOCX file
    """
    text = ""
    image_file_list = []
    
    try:
        # DOCX files are ZIP archives containing media files
        with zipfile.ZipFile(url) as z:
            all_files = z.namelist()
            images = sorted(filter(lambda x: x.startswith("word/media/"), all_files))
            
            # Extract each image from the DOCX
            for image in images:
                img_data = z.open(image).read()
                filename = os.path.basename(image)
                filepath = os.path.join(uploadpath, filename)
                
                with open(filepath, "wb") as f:
                    f.write(img_data)
                image_file_list.append(filepath)

        # Process each extracted image with OCR
        for image_file in image_file_list:
            text += gpt4o_ocr_text(image_file)
            # Clean up temporary image file
            os.remove(image_file)
            
    except Exception as e:
        st.error(f"Error extracting images from DOCX: {str(e)}")
        
    return text


def picurl2ocr(url):
    """
    Extract text from an image file using OCR.
    
    This is a simple wrapper function that calls gpt4o_ocr_text
    to process an image file with OCR.
    
    Args:
        url (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
    """
    return gpt4o_ocr_text(url)


def find_files(path: str, glob_pat: str, ignore_case: bool = False):
    """
    Find files matching a given glob pattern.
    
    Args:
        path (str): Directory path to search in
        glob_pat (str): Glob pattern to match filenames
        ignore_case (bool, optional): Whether to ignore case. Defaults to False.
        
    Returns:
        list: List of matching file paths
    """
    # Create regex pattern from glob pattern
    rule = (
        re.compile(fnmatch.translate(glob_pat), re.IGNORECASE)
        if ignore_case
        else re.compile(fnmatch.translate(glob_pat))
    )
    return [
        n for n in glob.glob(os.path.join(path, "*.*"), recursive=True) if rule.match(n)
    ]


def save_uploadedfile(uploadedfile, uploadpath):
    """
    Save an uploaded file to the specified path.
    
    Args:
        uploadedfile: Streamlit UploadedFile object
        uploadpath (str): Directory to save the file
        
    Returns:
        streamlit.delta_generator.DeltaGenerator: Success message
    """
    # Ensure upload directory exists
    os.makedirs(uploadpath, exist_ok=True)
    
    with open(os.path.join(uploadpath, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"File uploaded: {uploadedfile.name}")


def docxconvertion(uploadpath):
    """
    Convert doc, wps, and docx files to docx format using LibreOffice.
    
    This function finds all documents of supported types and converts them
    to docx format for further processing.
    
    Args:
        uploadpath (str): Base directory where files are stored
    """
    # Setup destination paths for converted files
    docdest = os.path.join(uploadpath, "doc")
    wpsdest = os.path.join(uploadpath, "wps")
    docxdest = os.path.join(uploadpath, "docx")
    
    # Create destination directories if they don't exist
    for directory in [docdest, wpsdest, docxdest]:
        os.makedirs(directory, exist_ok=True)
    
    # Find files of each type
    docfiles = find_files(uploadpath, "*.doc", True)
    wpsfiles = find_files(uploadpath, "*.wps", True)
    docxfiles = find_files(uploadpath, "*.docx", True)
    
    # Convert each file type using LibreOffice
    def convert_files(file_list, output_dir):
        for filepath in file_list:
            st.info(f"Converting {filepath}")
            subprocess.call([
                "soffice",
                "--headless",
                "--convert-to",
                "docx",
                filepath,
                "--outdir",
                output_dir,
            ])
    
    convert_files(docfiles, docdest)
    convert_files(wpsfiles, wpsdest)
    convert_files(docxfiles, docxdest)


def get_uploadfiles(uploadpath):
    """
    Get a list of all uploaded files in the specified directory.
    
    Args:
        uploadpath (str): Directory path to search for files
        
    Returns:
        list: List of filenames (base names only)
    """
    # Ensure the upload directory exists
    os.makedirs(uploadpath, exist_ok=True)
    
    fileslist = glob.glob(os.path.join(uploadpath, "*.*"), recursive=True)
    return [os.path.basename(file) for file in fileslist]


def remove_uploadfiles(uploadpath):
    """
    Remove all files in the upload directory and its subdirectories.
    
    This function cleans up temporary files and can be used to reset
    the application state.
    
    Args:
        uploadpath (str): Directory path to clean up
    """
    files = glob.glob(os.path.join(uploadpath, "**", "*.*"), recursive=True)
    
    for file in files:
        try:
            os.remove(file)
            st.info(f"Removed: {os.path.basename(file)}")
        except OSError as e:
            st.error(f"Error removing {file}: {e.strerror}")


def convert_uploadfiles(txtls, uploadpath):
    """
    Convert all files in the upload folder to text.
    
    This function processes each file based on its type and extracts text
    using the most appropriate method for that file type.
    
    Args:
        txtls (list): List of filenames to process
        uploadpath (str): Base directory where files are stored
        
    Returns:
        list: List of extracted text content for each file
    """
    resls = []
    for file in txtls:
        st.info(f"Processing: {file}")
        try:
            datapath = os.path.join(uploadpath, file)
            base, ext = os.path.splitext(file)

            # Handle different file types with appropriate extraction methods
            if ext.lower() == ".doc":
                datapath = os.path.join(uploadpath, "doc", base + ".docx")
                st.info(f"Using converted file: {datapath}")
                text = docxurl2txt(datapath)
                text1 = clean_string(text)
                if text1 == "":
                    text = docxurl2ocr(datapath, uploadpath)

            elif ext.lower() == ".wps":
                datapath = os.path.join(uploadpath, "wps", base + ".docx")
                st.info(f"Using converted file: {datapath}")
                text = docxurl2txt(datapath)
                text1 = clean_string(text)
                if text1 == "":
                    text = docxurl2ocr(datapath, uploadpath)

            elif ext.lower() == ".docx":
                st.info(f"Processing DOCX: {datapath}")
                text = docxurl2txt(datapath)
                text1 = clean_string(text)
                if text1 == "":
                    datapath = os.path.join(uploadpath, "docx", file)
                    st.info(f"Trying alternative path: {datapath}")
                    text = docxurl2txt(datapath)
                    text2 = clean_string(text)
                    if text2 == "":
                        text = docxurl2ocr(datapath, uploadpath)

            elif ext.lower() == ".pdf":
                text = pdfurl2txt(datapath)
                text1 = clean_string(text)
                if text1 == "":
                    st.info("Direct PDF text extraction failed, trying OCR...")
                    text = pdfurl2ocr(datapath, uploadpath)

            elif ext.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                st.info(f"Processing image with OCR: {file}")
                text = picurl2ocr(datapath)
            else:
                st.warning(f"Unsupported file format: {ext}")
                text = ""
        except Exception as e:
            st.error(f"Error processing file {file}: {str(e)}")
            text = ""
        resls.append(text)
    return resls


def extract_text(df, uploadpath):
    """
    Extract text from files listed in the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing file information
        uploadpath (str): Base directory where files are stored
        
    Returns:
        pandas.DataFrame: DataFrame with added text content column
    """
    txtls = df["文件"].tolist()
    resls = convert_uploadfiles(txtls, uploadpath)
    df["文本"] = resls
    return df


def picurl2table(url):
    """
    Extract table from an image using GPT-4o's vision capabilities.
    
    This function uses GPT-4o to directly extract tabular data from images
    without relying on traditional computer vision techniques.
    
    Args:
        url (str): Path to the image file
        
    Returns:
        pandas.DataFrame: Extracted table as a DataFrame
    """
    return gpt4o_extract_table(url)


def convert_tablefiles(txtls, uploadpath):
    """
    Convert all files in upload folder to tables.
    
    This function processes files that may contain tables and extracts
    the tabular data using either direct extraction or GPT-4o methods.
    
    Args:
        txtls (list): List of filenames to process
        uploadpath (str): Base directory where files are stored
        
    Returns:
        list: List of paths to saved CSV files containing extracted tables
    """
    resls = []
    for file in txtls:
        try:
            datapath = os.path.join(uploadpath, file)
            base, ext = os.path.splitext(file)
            
            if ext.lower() == ".pdf":
                st.info(f"Processing PDF for tables: {file}")
                # First try direct table extraction from PDF
                pdfls = pdf2table(datapath, uploadpath)
                if len(pdfls) > 0:
                    resls += pdfls
                else:
                    # If direct extraction fails, use OCR-based extraction
                    st.info("Direct PDF table extraction failed, trying GPT-4o...")
                    pdfls = pdfocr2table(datapath, uploadpath)
                    resls += pdfls
                    
            elif ext.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                st.info(f"Processing image for tables: {file}")
                # Use GPT-4o for table extraction from images
                tabledf = gpt4o_extract_table(datapath)
                
                # Display and save results
                if not tabledf.empty:
                    st.table(tabledf)
                    savename = os.path.join(uploadpath, base + ".csv")
                    tabledf.to_csv(savename, index=False)
                    st.success(f"Table saved to {savename}")
                    resls.append(savename)
                else:
                    st.warning(f"No tables found in {file}")
        except Exception as e:
            st.error(f"Error processing file {file} for tables: {str(e)}")
    return resls


def extract_table(df, uploadpath):
    """
    Extract tables from files listed in the DataFrame.
    
    This function processes each file in the DataFrame to extract
    tables and saves them as CSV files.
    
    Args:
        df (pandas.DataFrame): DataFrame containing file information
        uploadpath (str): Base directory where files are stored
        
    Returns:
        list: List of paths to saved CSV files containing extracted tables
    """
    txtls = df["文件"].tolist()
    resls = convert_tablefiles(txtls, uploadpath)
    return resls


def convert_df2zip(df, uploadpath):
    """
    Convert text extraction results to a ZIP file of text files.
    
    Args:
        df (pandas.DataFrame): DataFrame with file information and extracted text
        uploadpath (str): Directory to save the text files and ZIP file
        
    Returns:
        str: Path to the created ZIP file
    """
    filels = df["文件"].tolist()
    textls = df["文本"].tolist()
    
    # Save each text extraction to a separate text file
    for file, text in zip(filels, textls):
        base, ext = os.path.splitext(file)
        if ext.lower() != ".zip":
            filepath = os.path.join(uploadpath, base + ".txt")
            with open(filepath, "w") as f:
                f.write(text)
    
    # Get all text files in uploadpath
    txtfilels = glob.glob(os.path.join(uploadpath, "*.txt"))
    
    # Create ZIP file with all text files
    downloadname = os.path.join(uploadpath, "table_data.zip")
    with zipfile.ZipFile(downloadname, "w") as zf:
        for txtfile in txtfilels:
            zf.write(txtfile, os.path.basename(txtfile))
    
    return downloadname


def convert_table2zip(filels, uploadpath):
    """
    Create a ZIP file containing all table CSV files.
    
    Args:
        filels (list): List of CSV file paths
        uploadpath (str): Directory to save the ZIP file
        
    Returns:
        str: Path to the created ZIP file
    """
    downloadname = os.path.join(uploadpath, "table_data.zip")
    with zipfile.ZipFile(downloadname, "w") as zf:
        for file in filels:
            zf.write(file, os.path.basename(file))
    return downloadname


def pdfocr2table(url, uploadpath):
    """
    Extract tables from PDF by first converting pages to images then using GPT-4o.
    
    This is used when direct table extraction from a PDF fails. It converts
    each page to an image and then uses GPT-4o to identify and extract tables.
    
    Args:
        url (str): Path to the PDF file
        uploadpath (str): Directory to save temporary files and results
        
    Returns:
        list: List of paths to saved CSV files containing extracted tables
    """
    PDF_file = Path(url)
    # Get PDF basename
    pdfname = os.path.basename(url).split(".")[0]
    # Store all the pages of the PDF in a variable
    image_file_list = []
    resls = []
    
    # Convert PDF pages to images
    pdf_pages = convert_from_path(PDF_file, 500)
    
    # Save each page as an image
    for page_enumeration, page in enumerate(pdf_pages, start=1):
        filename = os.path.join(
            uploadpath, pdfname + "-page_" + str(page_enumeration) + ".jpg"
        )
        page.save(filename, "JPEG")
        image_file_list.append(filename)

    # Process each page image for tables
    for image_file in image_file_list:
        st.info(f"Processing page for tables: {os.path.basename(image_file)}")
        
        # Use GPT-4o to extract tables from the page image
        tabledf = gpt4o_extract_table(image_file)
        
        # Save results if a table was found
        if not tabledf.empty:
            st.table(tabledf)
            base, ext = os.path.splitext(image_file)
            savename = base + ".csv"
            tabledf.to_csv(savename, index=False)
            st.success(f"Table saved to {os.path.basename(savename)}")
            resls.append(savename)
        else:
            st.info(f"No tables found on page {os.path.basename(image_file)}")
            
        # Clean up temporary image file
        os.remove(image_file)

    return resls


def clean_string(string):
    """
    Remove all whitespace from a string.
    
    Used to check if extracted text is empty (contains only whitespace).
    
    Args:
        string (str): Input string to clean
        
    Returns:
        str: String with all whitespace removed
    """
    return re.sub(r"\s+", "", string)


def pdf2table(url, uploadpath):
    """
    Extract tables from PDF using pdfplumber's built-in table extraction.
    
    This attempts to directly extract tables from PDF without using GPT-4o,
    which is faster but may not work for all PDFs.
    
    Args:
        url (str): Path to the PDF file
        uploadpath (str): Directory to save extracted tables
        
    Returns:
        list: List of paths to saved CSV files containing extracted tables
    """
    # Get PDF basename
    pdfname = os.path.basename(url).split(".")[0]
    dfls = []
    
    try:
        with pdfplumber.open(url) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    if table is not None:
                        tabledf = pd.DataFrame(table)
                        # Display results
                        st.table(tabledf)
                        # Create a file name to store the table
                        filename = os.path.join(
                            uploadpath,
                            pdfname + "-page_" + str(i) + "_table_" + str(j) + ".csv",
                        )
                        tabledf.to_csv(filename, index=False)
                        st.success(f"Table saved to {os.path.basename(filename)}")
                        dfls.append(filename)
    except Exception as e:
        st.error(f"Error extracting tables directly from PDF: {str(e)}")
        
    return dfls
