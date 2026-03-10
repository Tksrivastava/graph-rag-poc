# import camelot

# tables = camelot.read_pdf(
#     "./dataset/tesla-report-10k.pdf",
#     pages="all",
#     flavor="stream"
# )

# df = tables[0].df
# print(df)

# for df in tables:
#     print(df.df)


from pypdf import PdfReader

reader = PdfReader("./dataset/tesla-report-10k.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

print(text)