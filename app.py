import pandas as pd
import streamlit as st

from doc2text import (
    convert_table2zip,
    docxconvertion,
    extract_table,
    extract_text,
    get_uploadfiles,
    remove_uploadfiles,
    save_uploadedfile,
)

uploadpath = "uploads/"


def main():
    st.subheader("文档转文本")

    menuls = ["文档上传", "文本抽取", "表格识别"]

    menu = st.sidebar.selectbox("选择菜单", menuls)

    if menu == "文档上传":

        uploaded_file_ls = st.file_uploader(
            "选择文档上传",
            type=["docx", "pdf", "doc", "wps", "bmp", "png", "jpg", "jpeg", "tiff"],
            accept_multiple_files=True,
            help="选择文档上传",
        )

        for uploaded_file in uploaded_file_ls:
            if uploaded_file is not None:
                save_uploadedfile(uploaded_file, uploadpath)

        # button for remove files
        remove_button = st.sidebar.button("文档删除")
        if remove_button:
            remove_uploadfiles(uploadpath)
        # display uploaded files
        filels = get_uploadfiles(uploadpath)
        if len(filels) > 0:
            # convert files to df
            df = pd.DataFrame({"文件": filels})
            st.write(df)
        else:
            st.error("请上传文档")
            st.stop()

    elif menu == "文本抽取":
        # display uploaded files
        filels = get_uploadfiles(uploadpath)
        if len(filels) > 0:
            # convert files to df
            df = pd.DataFrame({"文件": filels})
            st.write(df)
        else:
            st.error("请上传文档")
            st.stop()
        # button for convert
        convert_button = st.sidebar.button("word格式转换")
        if convert_button:
            docxconvertion(uploadpath)

        # button for convert
        convert_button = st.sidebar.button("文本抽取")
        if convert_button:
            dfnew = extract_text(df, uploadpath)
            st.table(dfnew)
            # add download button to left
            st.download_button(
                "下载结果", data=dfnew.to_csv().encode("utf_8_sig"), file_name="转换结果.csv"
            )

    elif menu == "表格识别":
        # display uploaded files
        filels = get_uploadfiles(uploadpath)
        if len(filels) > 0:
            # convert files to df
            df = pd.DataFrame({"文件": filels})
            st.write(df)
        else:
            st.error("请上传文档")
            st.stop()

        # button for convert
        convert_table = st.sidebar.button("表格识别")
        if convert_table:
            tablels = extract_table(df, uploadpath)
            downloadname = convert_table2zip(tablels, uploadpath)
            with open(downloadname, "rb") as f:
                st.download_button("下载结果", f, file_name="download.zip")


if __name__ == "__main__":
    main()
