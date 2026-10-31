"""Simple standalone HTML rendering for editable Markdown deliverables."""
from pathlib import Path
import re,html
ROOT=Path(__file__).resolve().parent

def inline(text):
    text=html.escape(text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
    return text

def render(path,out):
    lines=path.read_text(encoding='utf-8').splitlines();result=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));result.append(f'<h{level}>{inline(line[level:].strip())}</h{level}>');i+=1
        elif line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].startswith('|'):block.append(lines[i]);i+=1
            rows=[[cell.strip() for cell in row.strip('|').split('|')] for row in block]
            result.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>')
            for row in rows[2:]:result.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>')
            result.append('</tbody></table></div>')
        elif line.startswith('!['):
            m=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line)
            result.append(f'<figure><img src="{html.escape(m[2])}" alt="{html.escape(m[1])}"><figcaption>{html.escape(m[1])}</figcaption></figure>');i+=1
        elif line.startswith('- '):
            result.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):result.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            result.append('</ul>')
        else:
            parts=[]
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','- ')):parts.append(lines[i]);i+=1
            result.append('<p>'+inline(' '.join(parts))+'</p>')
    title=lines[0].lstrip('# ')
    out.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{font:16px/1.7 Georgia,serif;color:#183245;background:#eef3f5;margin:0}main{max-width:1000px;margin:30px auto;background:white;padding:65px 70px}h1,h2,h3,th{font-family:Segoe UI,Arial,sans-serif}h1{font-size:36px;line-height:1.2;margin:0 0 30px;color:#102b40}h2{font-size:25px;margin-top:45px;color:#087f8c}p{margin:14px 0}table{width:100%;border-collapse:collapse;font:14px/1.5 Segoe UI,sans-serif}th{text-align:left;background:#13364a;color:white}td,th{padding:11px 12px;border-bottom:1px solid #d8e2e7}tr:nth-child(even) td{background:#f4f7f9}.table-wrap{overflow:auto;margin:25px 0}code{font-size:13px;overflow-wrap:anywhere;background:#f0f4f6;padding:2px 4px}img{max-width:100%}figure{margin:30px 0}figcaption{font:13px Segoe UI,sans-serif;color:#627786}a{color:#087f8c}@media print{body{background:white}main{margin:0;padding:0;max-width:none}h2{break-after:avoid}figure,tr{break-inside:avoid}table{font-size:11px}h1{font-size:28px}}@media(max-width:700px){main{padding:30px 24px}h1{font-size:28px}}</style></head><body><main>'+''.join(result)+'</main></body></html>',encoding='utf-8')

if __name__=='__main__':
    render(ROOT/'deliverables/PROJECT_REPORT.md',ROOT/'deliverables/PROJECT_REPORT.html')
    render(ROOT/'VIVA_AND_CONTRIBUTIONS.md',ROOT/'deliverables/VIVA_GUIDE.html')
