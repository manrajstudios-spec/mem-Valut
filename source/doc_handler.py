import json
import time
import numpy as np
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

start_time = time.monotonic()
result = converter.convert("Data/doc_data/attention.pdf")
print(time.monotonic()-start_time)
    