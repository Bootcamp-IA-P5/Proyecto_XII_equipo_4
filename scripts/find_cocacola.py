from pathlib import Path
p=Path('data/output/yolo_dataset/labels')
if not p.exists():
    print('labels dir not found:', p)
    raise SystemExit
files_with=[]
ann_total=0
for f in p.rglob('*.txt'):
    try:
        with open(f,'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip().startswith('12 '):
                    files_with.append(str(f))
                    ann_total+=1
                    break
    except Exception as e:
        pass
print('files_with_class12:', len(files_with))
# Count total annotations with class 12 across all files
ann_total=0
for f in p.rglob('*.txt'):
    try:
        with open(f,'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip().startswith('12 '):
                    ann_total+=1
    except Exception:
        pass
print('total_annotations_with_class12:', ann_total)
for s in files_with[:10]:
    print(s)
