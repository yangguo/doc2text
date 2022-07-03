import pandas as pd
import streamlit as st

from doc2text import save_uploadedfile


def main():
    st.subheader("Convert Table to Word")

    uploaded_file_ls = st.file_uploader(
        "选择新文件上传",
        type=["docx", "pdf", "doc", "wps"],
        accept_multiple_files=True,
        help="选择文件上传",
    )

    for uploaded_file in uploaded_file_ls:
        if uploaded_file is not None:

            save_uploadedfile(uploaded_file)


if __name__ == "__main__":
    main()
