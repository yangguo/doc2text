import os

import streamlit as st

uploadfolder = "uploads"


def save_uploadedfile(uploadedfile):
    with open(os.path.join(uploadfolder, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("上传文件:{} 成功。".format(uploadedfile.name))
