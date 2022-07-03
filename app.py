import pandas as pd
import streamlit as st

from doc2text import (
    docxconvertion,
    extract_text,
    get_uploadfiles,
    remove_uploadfiles,
    save_uploadedfile,
)


def main():
    st.subheader("文档转文本")

    menuls = ["文档上传", "docx文件格式转换", "文本处理"]

    menu = st.sidebar.selectbox("选择菜单", menuls)

    if menu == "文档上传":

        uploaded_file_ls = st.file_uploader(
            "选择新文件上传",
            type=["docx", "pdf", "doc", "wps"],
            accept_multiple_files=True,
            help="选择文件上传",
        )

        for uploaded_file in uploaded_file_ls:
            if uploaded_file is not None:

                save_uploadedfile(uploaded_file)

    elif menu == "docx文件格式转换":
        # display uploaded files
        filels = get_uploadfiles()
        st.write(filels)

        # button for convert
        convert_button = st.sidebar.button("docx格式转换")
        if convert_button:
            docxconvertion()

        # button for remove files
        remove_button = st.sidebar.button("删除文件")
        if remove_button:
            remove_uploadfiles()

    elif menu == "文本处理":
        # display uploaded files
        filels = get_uploadfiles()
        # convert files to df
        df = pd.DataFrame({"文件": filels})
        st.write(df)

        # button for convert
        convert_button = st.sidebar.button("文件格式转换")
        if convert_button:
            dfnew = extract_text(df)
            st.table(dfnew)
            # add download button to left
            st.download_button(
                "下载结果", data=dfnew.to_csv().encode("utf_8_sig"), file_name="转换结果.csv"
            )


if __name__ == "__main__":
    main()
