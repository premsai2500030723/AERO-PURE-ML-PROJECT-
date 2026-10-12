import json
data = json.load(open('output/model_report/models.json'))
print(f'Models: {len(data)}')
for m in data:
    binacc = m.get('binary',{}).get('accuracy','—')
    mcacc  = m.get('multiclass',{}).get('accuracy','—')
    r2     = m.get('regression',{}).get('r2','—')
    print(f"  {m['id']:<36} r2={r2}  bin={binacc}  mc={mcacc}")
print()
for m in data:
    b = m.get('binary',{})
    if 'matrix' in b:
        s = sum(sum(r) for r in b['matrix'])
        print(f"  {m['id']} binary CM sum={s}")
    mc = m.get('multiclass',{})
    if 'matrix' in mc:
        s = sum(sum(r) for r in mc['matrix'])
        print(f"  {m['id']} 4class CM sum={s}")
