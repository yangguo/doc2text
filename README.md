# Doc2Text

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue)

> A powerful document text and table extraction tool powered by GPT-4o

## 📝 Description

Doc2Text is an intelligent document processing application that leverages OpenAI's GPT-4o vision capabilities to extract both text and tables from various document formats with high accuracy. Perfect for converting documents to machine-readable text or extracting structured data from tables.

### 🔍 Key Features

- **Universal Document Support**: Handles multiple formats including PDF, DOCX, DOC, WPS, and various image formats (PNG, JPG, TIFF, etc.)
- **Smart Text Extraction**: Uses GPT-4o for accurate OCR on documents with complex layouts
- **Advanced Table Recognition**: Extract tables from documents directly using GPT-4o's vision capabilities
- **Easy-to-use Interface**: Simple Streamlit web interface for uploading documents and downloading results
- **Batch Processing**: Process multiple documents at once

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- OpenAI API key (for GPT-4o access)
- LibreOffice (for DOC/WPS format conversion)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/doc2text.git
cd doc2text
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. For document format conversion support, install LibreOffice:
   - macOS: `brew install --cask libreoffice`
   - Ubuntu: `sudo apt install libreoffice`

4. Create a `.env` file in the project directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## 🖥 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then follow these steps in the web interface:

1. **Upload Documents**: Select "文档上传" (Document Upload) from the sidebar menu and upload one or more supported documents
2. **Extract Text**: Choose "文本抽取" (Text Extraction) to process documents and extract their text content
3. **Extract Tables**: Select "表格识别" (Table Recognition) to identify and extract tables from documents

Results can be downloaded as ZIP files containing text or CSV files.

## 🛠 Architecture

Doc2Text uses a pipeline approach to document processing:

1. **Document Upload**: Files are stored in the `uploads/` directory
2. **Format Conversion**: Non-compatible formats are converted to processable formats
3. **Text/Table Extraction**: GPT-4o is used to extract text or table data from the documents
4. **Result Packaging**: Extracted content is saved and packaged for download

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---
title: Doc2text
emoji: 🦀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.10.0
app_file: app.py
pinned: false
license: apache-2.0
---
