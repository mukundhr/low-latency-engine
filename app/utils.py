import re


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(" . ", ".\n")
    text = text.replace(" - ", "\n- ")
    return text.strip()


def split_into_paragraphs(text):
    # split by sections
    sections = re.split(r'\n\s*\n|---', text)

    processed = []

    for sec in sections:
        sec = sec.strip()

        # keep numbered lists intact
        if re.search(r'\d+\.', sec):
            processed.append(sec)
        elif len(sec) > 50:
            processed.append(sec)

    return processed


def chunk_paragraph(paragraph, chunk_size=250, overlap=40):
    chunks = []
    start = 0

    while start < len(paragraph):
        end = start + chunk_size
        chunk = paragraph[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_text(text):
    text = clean_text(text)
    paragraphs = split_into_paragraphs(text)

    chunks = []

    for para in paragraphs:
        # if it's a structured list → keep whole
        if re.search(r'\d+\.', para):
            chunks.append(para)
        elif len(para) <= 300:
            chunks.append(para)
        else:
            chunks.extend(chunk_paragraph(para))

    return chunks