import pdfplumber
import pdf2image
import re
from typing import List, Tuple
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os


class PDFProcessor:
    def __init__(self):
        pass

    def extract_text_with_page_numbers(self, pdf) -> Tuple[str, List[Tuple[str, int]]]:
        """
        从PDF中提取文本并记录每个字符对应的页码

        参数:
            pdf: PDF文件对象

        返回:
            text: 提取的文本内容
            char_page_mapping: 每个字符对应的页码列表
        """
        text = ""
        char_page_mapping = []

        for page_number, page in enumerate(pdf.pages, start=1):
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
                # 为当前页面的每个字符记录页码
                char_page_mapping.extend([page_number] * len(extracted_text))
            else:
                print(f"No text found on page {page_number}.")

        return text, char_page_mapping

    def image_to_text(self, image_path):
        """对图片进行OCR和CLIP描述。"""
        try:
            image = Image.open(image_path)
            # OCR
            ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng').strip()
            return {"ocr": ocr_text}
        except Exception as e:
            print(f"处理图片失败 {image_path}: {e}")
            return {"ocr": ""}

    def images_to_text(self, img_dir):
        pages_text = []
        for img_filename in os.listdir(img_dir):
            if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(img_dir, img_filename)
                print(f"    - 处理图片: {img_filename}")

                img_text_info = self.image_to_text(img_path)
                pages_text.append(img_text_info["ocr"])

        return pages_text

    def pdf_to_text(self, pdf_path: str, dpi: int = 300):
        # 1. 将PDF转换为图像列表
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=r'D:\aiPackage\poppler-25.07.0\Library\bin')

        all_text = []
        for i, image in enumerate(images):
            # 2. 图像预处理（示例：转灰度+二值化）
            gray = image.convert('L')  # 转灰度
            # 使用阈值二值化 (调整阈值以适应你的文档)
            threshold = 140
            binary = gray.point(lambda p: p > threshold and 255)

            # 3. 使用Tesseract进行OCR
            text = pytesseract.image_to_string(binary, lang='eng+chi_sim')  # 中英文识别
            all_text.append(text)

        return "\n\n".join(all_text)  # 用空行分隔不同页

    def extract_text(self, pdf_path: str) -> List[str]:
        """从PDF提取文本"""
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # 基础文本清理
                    text = re.sub(r'\s+', ' ', text)
                    pages_text.append(text)
        return pages_text

    def extract_text_with_ocr(self, pdf_path: str) -> List[str]:
        """使用OCR从PDF提取文本（适用于扫描版PDF）"""
        pages_text = []
        try:
            # 将PDF转换为图片
            pages = convert_from_path(pdf_path)
            for page_image in pages:
                # 使用OCR提取文本
                text = pytesseract.image_to_string(page_image, lang='chi_sim')
                if text.strip():
                    text = re.sub(r'\s+', ' ', text).strip()
                    pages_text.append(text)
        except Exception as e:
            print(f"OCR提取文本时出错: {e}")

        return pages_text


    def extract_tables(self, pdf_path: str):
        """提取表格数据（膳食指南中的表格很重要）"""
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and any(any(cell for cell in row) for row in table):
                        tables.append(table)
        return tables