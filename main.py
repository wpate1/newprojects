import fileinput
import torch
import pdfplumber
import sentencepiece
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage



app = Flask(__name__)


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'


with pdfplumber.open(fileinput) as pdf:
    extracted_page = pdf.pages[1]
    extracted_text = extracted_page.extract_text()
    print(extracted_text)

model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

inputs = tokenizer([extracted_text], truncation=True, return_tensors='pt')

# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, early_stopping=True, min_length=0, max_length=1024)
summarized_text = (
[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])
print(summarized_text[0])

if __name__ == '__main__':
    app.run(debug=True)
