import re

with open("marker_output/extracted_text.md", encoding="utf-8") as f:
    text = f.read()

headers = re.findall(r'^(#{1,4})\s*(.+)$', text, re.MULTILINE)

# نكتب النتيجة مباشرة لملف من داخل بايثون - نتحكم بالترميز بأنفسنا
# utf-8-sig يضيف BOM (علامة تعريف) في بداية الملف عشان أي برنامج
# (بما فيه Notepad) يتعرف صح إنه UTF-8 تلقائياً
with open("headers_audit.txt", "w", encoding="utf-8-sig") as out:
    out.write(f"إجمالي عدد العناوين: {len(headers)}\n\n")
    for level, title in headers:
        out.write(f"{level} {title.strip()}\n")

print(f"تم! النتيجة محفوظة في headers_audit.txt ({len(headers)} عنوان)")