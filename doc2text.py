import fnmatch
import glob
import os
import re
import subprocess
import zipfile
from pathlib import Path

import cv2
import docx
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


ocr = PaddleOCR(use_angle_cls=True, lang="ch")


def docxurl2txt(url):

    text = ""
    try:
        doc = docx.Document(url)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
            text = "\n".join(fullText)
    except Exception as e:
        st.error(str(e))

    return text


def pdfurl2txt(url):
    #     response = requests.get(url)
    #     source_stream = BytesIO(response.content)
    result = ""
    try:
        #         with pdfplumber.open(source_stream) as pdf:
        with pdfplumber.open(url) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt != "":
                    result += txt
    except Exception as e:
        st.error(str(e))
    return result


def paddleocr2text(image_file):
    result = ocr.ocr(image_file, cls=True)
    text = ""
    for idx in range(len(result)):
        res = result[idx]
        txts = [line[1][0] for line in res]
        text += "\n".join(txts)
    return text


def pdfurl2ocr(url, uploadpath):
    PDF_file = Path(url)
    # Store all the pages of the PDF in a variable
    image_file_list = []
    text = ""
    # with TemporaryDirectory() as tempdir:
    pdf_pages = convert_from_path(PDF_file, 500)
    # Iterate through all the pages stored above
    for page_enumeration, page in enumerate(pdf_pages, start=1):
        # enumerate() "counts" the pages for us.

        # Create a file name to store the image
        filename = os.path.join(uploadpath, "page_" + str(page_enumeration) + ".jpg")

        # Save the image of the page in system
        page.save(filename, "JPEG")
        image_file_list.append(filename)

    # Iterate from 1 to total number of pages
    for image_file in image_file_list:
        text += paddleocr2text(image_file)
        # delete image file
        os.remove(image_file)

    return text


def docxurl2ocr(url, uploadpath):
    z = zipfile.ZipFile(url)
    all_files = z.namelist()
    images = sorted(filter(lambda x: x.startswith("word/media/"), all_files))

    # Store all the pages of the PDF in a variable
    image_file_list = []
    text = ""
    # with TemporaryDirectory() as tempdir:
    # Iterate through all the pages stored above
    for page_enumeration, image in enumerate(images):
        # enumerate() "counts" the pages for us.
        img = z.open(image).read()
        # Create a file name to store the image
        filename = os.path.basename(image)
        filepath = os.path.join(uploadpath, filename)
        #             print(filename)
        # Save the image of the page in system
        f = open(filepath, "wb")
        f.write(img)
        image_file_list.append(filepath)

    # Iterate from 1 to total number of pages
    for image_file in image_file_list:
        text += paddleocr2text(image_file)
        # delete image file
        os.remove(image_file)

    return text


def picurl2ocr(url):
    text = ""
    text += paddleocr2text(url)
    return text


def find_files(path: str, glob_pat: str, ignore_case: bool = False):
    rule = (
        re.compile(fnmatch.translate(glob_pat), re.IGNORECASE)
        if ignore_case
        else re.compile(fnmatch.translate(glob_pat))
    )
    return [
        n for n in glob.glob(os.path.join(path, "*.*"), recursive=True) if rule.match(n)
    ]


def save_uploadedfile(uploadedfile, uploadpath):
    with open(os.path.join(uploadpath, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("上传文件:{} 成功。".format(uploadedfile.name))


def docxconvertion(uploadpath):

    docdest = os.path.join(uploadpath, "doc")
    wpsdest = os.path.join(uploadpath, "wps")
    # doccdest = os.path.join(basepath,'docc')
    docxdest = os.path.join(uploadpath, "docx")

    docfiles = find_files(uploadpath, "*.doc", True)
    wpsfiles = find_files(uploadpath, "*.wps", True)
    docxfiles = find_files(uploadpath, "*.docx", True)

    for filepath in docfiles:
        st.info(filepath)
        # filename = os.path.basename(filepath)
        #     print(filename)
        #         output = subprocess.check_output(["soffice","--headless","--convert-to","docx",file,"--outdir",dest])
        subprocess.call(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "docx",
                filepath,
                "--outdir",
                docdest,
            ]
        )

    for filepath in wpsfiles:
        st.info(filepath)
        # filename = os.path.basename(filepath)
        #     print(filename)
        #         output = subprocess.check_output(["soffice","--headless","--convert-to","docx",file,"--outdir",dest])
        subprocess.call(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "docx",
                filepath,
                "--outdir",
                wpsdest,
            ]
        )

    # for filepath in doccfiles:
    #     print (filepath)
    #     filename=os.path.basename(filepath)
    # #     print(filename)
    # #         output = subprocess.check_output(["soffice","--headless","--convert-to","docx",file,"--outdir",dest])
    #     subprocess.call(['soffice', '--headless', '--convert-to', 'docx', filepath,"--outdir",doccdest])

    for filepath in docxfiles:
        st.info(filepath)
        # filename = os.path.basename(filepath)
        #     print(filename)
        #         output = subprocess.check_output(["soffice","--headless","--convert-to","docx",file,"--outdir",dest])
        subprocess.call(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "docx",
                filepath,
                "--outdir",
                docxdest,
            ]
        )


def get_uploadfiles(uploadpath):
    fileslist = glob.glob(uploadpath + "/*.*", recursive=True)
    basenamels = []
    for file in fileslist:
        basenamels.append(os.path.basename(file))
    return basenamels


def remove_uploadfiles(uploadpath):
    files = glob.glob(uploadpath + "**/*.*", recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            st.error("Error: %s : %s" % (f, e.strerror))


# convert all files in uploadfolder to text
def convert_uploadfiles(txtls, uploadpath):

    resls = []
    for file in txtls:
        st.info(file)
        try:
            #     datapath=filepath+file
            datapath = os.path.join(uploadpath, file)
            #     get file ext
            base, ext = os.path.splitext(file)

            if ext.lower() == ".doc":
                datapath = os.path.join(uploadpath, "doc", base + ".docx")
                st.info(datapath)
                text = docxurl2txt(datapath)
                text1 = clean_string(text)
                if text1 == "":
                    text = docxurl2ocr(datapath, uploadpath)

            elif ext.lower() == ".wps":
                datapath = os.path.join(uploadpath, "wps", base + ".docx")
                st.info(datapath)
                text = docxurl2txt(datapath)
                text1 = clean_string(text)
                if text1 == "":
                    text = docxurl2ocr(datapath, uploadpath)

            #         elif ext.lower()=='doc.docx':
            #             datapath=os.path.join(filepath,'docc',file)
            #             print(datapath)
            #             text=docxurl2txt(datapath)
            elif ext.lower() == ".docx":
                st.info(datapath)
                text = docxurl2txt(datapath)
                text1 = clean_string(text)
                if text1 == "":
                    datapath = os.path.join(uploadpath, "docx", file)
                    st.info(datapath)
                    text = docxurl2txt(datapath)
                    text2 = clean_string(text)
                    if text2 == "":
                        text = docxurl2ocr(datapath, uploadpath)

            elif ext.lower() == ".pdf":
                text = pdfurl2txt(datapath)
                text1 = clean_string(text)
                if text1 == "":
                    text = pdfurl2ocr(datapath, uploadpath)

            elif (
                ext.lower() == ".png"
                or ext.lower() == ".jpg"
                or ext.lower() == ".jpeg"
                or ext.lower() == ".bmp"
                or ext.lower() == ".tiff"
            ):
                text = picurl2ocr(datapath)
            else:
                text = ""
        except Exception as e:
            st.error(str(e))
            text = ""
        resls.append(text)
    return resls


# extract text from files
def extract_text(df, uploadpath):
    txtls = df["文件"].tolist()
    resls = convert_uploadfiles(txtls, uploadpath)
    df["文本"] = resls
    return df


def seg_pic(img):
    image = cv2.imread(img, 1)

    # 灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    binary = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5
    )
    # ret,binary = cv2.threshold(~gray, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("二值化图片：", binary)  # 展示图片
    # cv2.waitKey(0)

    rows, cols = binary.shape
    scale = 40
    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    # cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("表格横线展示：", dilatedcol)
    # cv2.waitKey(0)

    # 识别竖线
    scale = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("表格竖线展示：", dilatedrow)
    # cv2.waitKey(0)

    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
    # cv2.imshow("表格交点展示：", bitwiseAnd)
    # cv2.waitKey(0)
    # cv2.imwrite("my.png",bitwiseAnd) #将二值像素点生成图片保存

    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    # cv2.imshow("表格整体展示：", merge)
    # cv2.waitKey(0)

    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(binary, merge)
    # cv2.imshow("图片去掉表格框线展示：", merge2)
    # cv2.waitKey(0)

    # 识别黑白图中的白色交叉点，将横纵坐标取出
    ys, xs = np.where(bitwiseAnd > 0)

    mylisty = []  # 纵坐标
    mylistx = []  # 横坐标

    # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点
    # 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
    i = 0
    myxs = np.sort(xs)
    for i in range(len(myxs) - 1):
        if myxs[i + 1] - myxs[i] > 10:
            mylistx.append(myxs[i])
        i = i + 1
    mylistx.append(myxs[i])  # 要将最后一个点加入

    i = 0
    myys = np.sort(ys)
    # print(np.sort(ys))
    for i in range(len(myys) - 1):
        if myys[i + 1] - myys[i] > 10:
            mylisty.append(myys[i])
        i = i + 1
    mylisty.append(myys[i])  # 要将最后一个点加入
    return image, mylistx, mylisty


def table_ocr(image, mylistx, mylisty):
    # ocr = PaddleOCR(det=True)
    # 循环y坐标，x坐标分割表格
    mylist = []
    for i in range(len(mylisty) - 1):
        row = []
        for j in range(len(mylistx) - 1):
            # 在分割时，第一个参数为y坐标，第二个参数为x坐标
            ROI = image[
                mylisty[i] + 3 : mylisty[i + 1] - 3, mylistx[j] : mylistx[j + 1] - 3
            ]  # 减去3的原因是由于我缩小ROI范围
            # cv2.imshow("分割后子图片展示：", ROI)
            # cv2.waitKey(0)
            result = ocr.ocr(ROI, det=True)
            text_len = len(result)
            tmptxt = " "
            txt = " "
            if text_len != 0:
                text = ""
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        tmptxt, _ = line[-1]
                        txt = txt + "\n" + tmptxt
                    text += txt
            row.append(text)
            j = j + 1
        i = i + 1
        mylist.append(row)

    return mylist


# convert all files in uploadfolder to table
def convert_tablefiles(txtls, uploadpath):

    resls = []
    for file in txtls:
        try:
            #     datapath=filepath+file
            datapath = os.path.join(uploadpath, file)
            #     get file ext
            base, ext = os.path.splitext(file)

            if ext.lower() == ".pdf":
                st.info(file)
                pdfls = pdf2table(datapath, uploadpath)
                if len(pdfls) > 0:
                    resls += pdfls
                else:
                    pdfls = pdfocr2table(datapath, uploadpath)
                    resls += pdfls
            elif (
                ext.lower() == ".png"
                or ext.lower() == ".jpg"
                or ext.lower() == ".jpeg"
                or ext.lower() == ".bmp"
                or ext.lower() == ".tiff"
            ):
                st.info(file)
                tabledf = picurl2table(datapath)
                # display results
                st.table(tabledf)
                # save file
                savename = os.path.join(uploadpath, base + ".csv")
                tabledf.to_csv(savename, index=False)
                resls.append(savename)
        except Exception as e:
            st.error(str(e))
    return resls


def picurl2table(url):
    image, mylistx, mylisty = seg_pic(url)
    mylist = table_ocr(image, mylistx, mylisty)
    df = pd.DataFrame(mylist)
    return df


# extract text from files
def extract_table(df, uploadpath):
    txtls = df["文件"].tolist()
    resls = convert_tablefiles(txtls, uploadpath)
    return resls


# convert dataframe to csv zipfile
def convert_df2zip(df, uploadpath):
    filels = df["文件"].tolist()
    textls = df["文本"].tolist()
    for file, text in zip(filels, textls):
        base, ext = os.path.splitext(file)
        if ext.lower() != ".zip":
            filepath = os.path.join(uploadpath, base + ".txt")
            with open(filepath, "w") as f:
                f.write(text)
    # get text file list in uploadpath
    txtfilels = glob.glob(os.path.join(uploadpath, "*.txt"))

    downloadname = os.path.join(uploadpath, "table_data.zip")
    with zipfile.ZipFile(downloadname, "w") as zf:
        for txtfile in txtfilels:
            zf.write(txtfile, os.path.basename(txtfile))
    return downloadname


# convert table files to csv zipfile
def convert_table2zip(filels, uploadpath):
    downloadname = os.path.join(uploadpath, "table_data.zip")
    with zipfile.ZipFile(downloadname, "w") as zf:
        for file in filels:
            zf.write(file, os.path.basename(file))
    return downloadname


def pdfocr2table(url, uploadpath):
    PDF_file = Path(url)
    # get url basename
    pdfname = os.path.basename(url).split(".")[0]
    # Store all the pages of the PDF in a variable
    image_file_list = []
    resls = []
    # with TemporaryDirectory() as tempdir:
    pdf_pages = convert_from_path(PDF_file, 500)
    # Iterate through all the pages stored above
    for page_enumeration, page in enumerate(pdf_pages, start=1):
        # enumerate() "counts" the pages for us.

        # Create a file name to store the image
        filename = os.path.join(
            uploadpath, pdfname + "-page_" + str(page_enumeration) + ".jpg"
        )

        # Save the image of the page in system
        page.save(filename, "JPEG")
        image_file_list.append(filename)

    # Iterate from 1 to total number of pages
    for image_file in image_file_list:
        tabledf = picurl2table(image_file)
        # display results
        st.table(tabledf)
        # get basename
        base, ext = os.path.splitext(image_file)
        # save file
        savename = base + ".csv"
        tabledf.to_csv(savename, index=False)
        resls.append(savename)
        # delete image file
        os.remove(image_file)

    return resls


# remove all spaces in string
def clean_string(string):
    return re.sub(r"\s+", "", string)


def pdf2table(url, uploadpath):
    # get url basename
    pdfname = os.path.basename(url).split(".")[0]
    dfls = []
    with pdfplumber.open(url) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                if table is not None:
                    tabledf = pd.DataFrame(table)
                    # display results
                    st.table(tabledf)
                    # Create a file name to store the image
                    filename = os.path.join(
                        uploadpath,
                        pdfname + "-page_" + str(i) + "_no_" + str(j) + ".csv",
                    )
                    tabledf.to_csv(filename, index=False)
                    dfls.append(filename)
    return dfls
