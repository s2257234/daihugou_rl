import joblib, os, numpy as np
path = "replay_buffer.joblib"
if not os.path.exists(path):
    print("replay_buffer.joblib が存在しません。")
    raise SystemExit(0)
rb = joblib.load(path)
def is_val(s):
    try:
        return isinstance(s, dict) and s.get('split') == 'val'
    except Exception:
        return False
vals = [s for s in rb if is_val(s)] if isinstance(rb, list) else []
print("total_samples:", len(rb) if hasattr(rb,'__len__') else type(rb))
print("val_samples:", len(vals))
# 最初の val サンプルの state と hand_labels を表示
if vals:
    st = vals[0].get('state', {})
    print("state keys:", list(st.keys()))
    hl = st.get('hand_labels', None)
    print("hand_labels present:", hl is not None)
    if hl is not None:
        a = np.asarray(list(hl), dtype=np.float32)
        print("hand_labels shape:", a.shape, "positive_count:", int(a.sum()))
else:
    print("val サンプルが見つかりません。")
