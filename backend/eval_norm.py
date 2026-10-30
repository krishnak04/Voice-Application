import json, re, sys
sys.path.insert(0, 'backend')
from benchmark_accuracy import calculate_wer, calculate_cer

def normalize_text(text: str) -> str:
    t = text.lower()
    t = t.replace('$', ' dollars ')
    t = t.replace('%', ' percent ')
    t = re.sub(r'(\d+):(\d+)', r'\1 \2', t)
    num_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
        '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
        '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
        '80': 'eighty', '90': 'ninety', '100': 'one hundred',
        '145': 'one hundred and forty five', '2024': 'twenty twenty four'
    }
    for digit, word in sorted(num_map.items(), key=lambda x: -len(x[0])):
        t = re.sub(r'\b' + digit + r'\b', word, t)
    t = re.sub(r'[^\w\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

with open('backend/accuracy_benchmark_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for model_name, res in data.items():
    raw_wers, norm_wers, raw_cers, norm_cers = [], [], [], []
    for s in res['samples']:
        ref, hyp = s['reference'], s['hypothesis']
        rw = s['wer']
        raw_wers.append(rw)
        n_ref, n_hyp = normalize_text(ref), normalize_text(hyp)
        nw = calculate_wer(n_ref, n_hyp)
        rc = calculate_cer(ref, hyp)
        nc = calculate_cer(n_ref, n_hyp)
        norm_wers.append(nw)
        raw_cers.append(rc)
        norm_cers.append(nc)
    avg_lat = res.get('average_latency_sec', 0.0)
    print(f"{model_name}:")
    print(f"   Raw  WER: {sum(raw_wers)/len(raw_wers)*100:5.2f}% | Norm WER: {sum(norm_wers)/len(norm_wers)*100:5.2f}%")
    print(f"   Raw  CER: {sum(raw_cers)/len(raw_cers)*100:5.2f}% | Norm CER: {sum(norm_cers)/len(norm_cers)*100:5.2f}%")
    print(f"   Exact Match: {sum(1 for w in norm_wers if w == 0.0)}/50 ({sum(1 for w in norm_wers if w == 0.0)/50*100:.1f}%)")
    print(f"   Avg Latency: {avg_lat:.3f}s")
    
    # Accent breakdown
    accents = {}
    for s in res['samples']:
        acc = s.get('accent', 'Unknown')
        if acc not in accents:
            accents[acc] = []
        n_ref, n_hyp = normalize_text(s['reference']), normalize_text(s['hypothesis'])
        accents[acc].append(calculate_wer(n_ref, n_hyp))
    acc_str = ", ".join([f"{a}: {sum(w_list)/len(w_list)*100:.1f}%" for a, w_list in accents.items()])
    print(f"   Accents Norm WER -> {acc_str}\n")
