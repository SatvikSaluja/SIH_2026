"""Minimal, dependable Markdown-subset -> PDF renderer (reportlab + DejaVu)."""
import re, sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageTemplate,
                               Paragraph, Preformatted, Spacer, Table, TableStyle)

D = "/usr/share/fonts/truetype/dejavu/"
for name, f in [("DJ", "DejaVuSans.ttf"), ("DJ-B", "DejaVuSans-Bold.ttf"),
                ("DJ-I", "DejaVuSerif.ttf"), ("DJM", "DejaVuSansMono.ttf"),
                ("DJM-B", "DejaVuSansMono-Bold.ttf")]:
    pdfmetrics.registerFont(TTFont(name, D + f))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I", boldItalic="DJ-B")

INK, MUTED, RULE = colors.HexColor("#15191f"), colors.HexColor("#5c6673"), colors.HexColor("#d8dee6")
ACCENT, BG = colors.HexColor("#1a4d8f"), colors.HexColor("#f4f6f9")
OK, BAD = colors.HexColor("#1d6b3f"), colors.HexColor("#9b2c2c")

S = {
 "title": ParagraphStyle("title", fontName="DJ-B", fontSize=21, leading=26, textColor=INK, spaceAfter=3),
 "sub":   ParagraphStyle("sub", fontName="DJ", fontSize=10.5, leading=15, textColor=MUTED, spaceAfter=14),
 "h1":    ParagraphStyle("h1", fontName="DJ-B", fontSize=15, leading=19, textColor=INK, spaceBefore=18, spaceAfter=7),
 "h2":    ParagraphStyle("h2", fontName="DJ-B", fontSize=11.8, leading=16, textColor=ACCENT, spaceBefore=13, spaceAfter=5),
 "body":  ParagraphStyle("body", fontName="DJ", fontSize=9.6, leading=14.6, textColor=INK, spaceAfter=7, alignment=TA_JUSTIFY),
 "li":    ParagraphStyle("li", fontName="DJ", fontSize=9.6, leading=14.4, textColor=INK,
                         leftIndent=13, bulletIndent=3, spaceAfter=4, alignment=TA_JUSTIFY),
 "cell":  ParagraphStyle("cell", fontName="DJ", fontSize=8.3, leading=11.4, textColor=INK),
 "cellh": ParagraphStyle("cellh", fontName="DJ-B", fontSize=8.3, leading=11.4, textColor=colors.white),
 "note":  ParagraphStyle("note", fontName="DJ-I", fontSize=8.8, leading=13, textColor=MUTED, spaceAfter=8),
}

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def inline(t):
    t = esc(t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 <font name='DJM' size=7.4 color='#5c6673'>(\2)</font>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<font name='DJM' size=8.6 color='#9b2c2c'>\1</font>", t)
    return t

def table(rows, width):
    head, body = rows[0], rows[1:]
    n = len(head)
    if n == 2:   cw = [width*0.34, width*0.66]
    elif n == 3: cw = [width*0.24, width*0.38, width*0.38]
    else:        cw = [width/n]*n
    data = [[Paragraph(inline(c), S["cellh"]) for c in head]]
    data += [[Paragraph(inline(c), S["cell"]) for c in r] for r in body]
    t = Table(data, colWidths=cw, repeatRows=1, hAlign="LEFT")
    st = [("BACKGROUND",(0,0),(-1,0),ACCENT), ("VALIGN",(0,0),(-1,-1),"TOP"),
          ("GRID",(0,0),(-1,-1),0.4,RULE), ("TOPPADDING",(0,0),(-1,-1),5),
          ("BOTTOMPADDING",(0,0),(-1,-1),5), ("LEFTPADDING",(0,0),(-1,-1),6),
          ("RIGHTPADDING",(0,0),(-1,-1),6)]
    for i in range(1, len(data)):
        if i % 2 == 0: st.append(("BACKGROUND",(0,i),(-1,i),BG))
    t.setStyle(TableStyle(st))
    return t

def build(md, out, title, subtitle):
    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=19*mm, rightMargin=19*mm,
                          topMargin=17*mm, bottomMargin=17*mm, title=title, author="GeoCadastra review")
    W = doc.width
    def deco(cv, d):
        cv.saveState(); cv.setFont("DJ", 7.6); cv.setFillColor(MUTED)
        cv.drawString(19*mm, 10*mm, title)
        cv.drawRightString(A4[0]-19*mm, 10*mm, "Page %d" % cv.getPageNumber())
        cv.setStrokeColor(RULE); cv.setLineWidth(0.4)
        cv.line(19*mm, 13*mm, A4[0]-19*mm, 13*mm); cv.restoreState()
    doc.addPageTemplates([PageTemplate(id="n", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=deco)])

    flow = [Paragraph(esc(title), S["title"]), Paragraph(esc(subtitle), S["sub"])]
    lines, i = md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            buf, i = [], i+1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            p = Preformatted("\n".join(buf), ParagraphStyle("c", fontName="DJM", fontSize=7.9, leading=11,
                             textColor=INK, backColor=BG, borderPadding=6, leftIndent=2))
            flow += [Spacer(1,3), p, Spacer(1,8)]; continue
        if ln.strip().startswith("|") and i+1 < len(lines) and set(lines[i+1].replace("|","").strip()) <= set("-: "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "): rows.append(cells)
                i += 1
            flow += [Spacer(1,3), table(rows, W), Spacer(1,10)]; continue
        s = ln.strip()
        if not s: flow.append(Spacer(1,2)); i += 1; continue
        if s.startswith("### "): flow.append(Paragraph(inline(s[4:]), S["h2"]))
        elif s.startswith("## "): flow.append(Paragraph(inline(s[3:]), S["h1"]))
        elif s.startswith("# "):  flow.append(Paragraph(inline(s[2:]), S["h1"]))
        elif re.match(r"^\*\*[^*]+\*\*$", s): flow.append(Paragraph(inline(s)[3:-4], S["h1"]))
        elif re.match(r"^\d+\.\s", s):
            num, rest = s.split(".", 1)
            flow.append(Paragraph(inline(rest.strip()), S["li"], bulletText=num + "."))
        elif s.startswith("- "): flow.append(Paragraph(inline(s[2:]), S["li"], bulletText="•"))
        elif s.startswith("> "): flow.append(Paragraph(inline(s[2:]), S["note"]))
        else: flow.append(Paragraph(inline(s), S["body"]))
        i += 1
    doc.build(flow)

if __name__ == "__main__":
    src, out, title, sub = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    build(open(src, encoding="utf-8").read(), out, title, sub)
    print("wrote", out)
