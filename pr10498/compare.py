"""Per-frame PSNR/SSIM + audio comparison between two FastH3-Omni outputs, with plots."""
import json, sys
import av, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    with av.open(path) as m:
        frames = [f.to_ndarray(format="rgb24") for f in m.decode(video=0)]
    with av.open(path) as m:
        audio = np.concatenate([f.to_ndarray() for f in m.decode(audio=0)], axis=1)
    return np.stack(frames), audio

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

def ssim(a, b):
    # global-window SSIM on luma, 8x8 block average
    def luma(x): return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)
    x, y = luma(a), luma(b)
    h, w = (x.shape[0] // 8) * 8, (x.shape[1] // 8) * 8
    x = x[:h, :w].reshape(h // 8, 8, w // 8, 8); y = y[:h, :w].reshape(h // 8, 8, w // 8, 8)
    mx, my = x.mean((1, 3)), y.mean((1, 3))
    vx, vy = x.var((1, 3)), y.var((1, 3))
    cxy = ((x - mx[:, None, :, None]) * (y - my[:, None, :, None])).mean((1, 3))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())

pairs = json.loads(sys.argv[1])  # [[label, a.mp4, b.mp4], ...]
out = sys.argv[2]
res = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for label, pa, pb in pairs:
    fa, aa = load(pa); fb, ab = load(pb)
    n = min(len(fa), len(fb))
    p = [psnr(fa[i], fb[i]) for i in range(n)]
    s = [ssim(fa[i], fb[i]) for i in range(n)]
    m = min(aa.shape[1], ab.shape[1])
    ad = aa[:, :m].astype(np.float64) - ab[:, :m].astype(np.float64)
    asnr = 10 * np.log10(np.sum(aa[:, :m].astype(np.float64) ** 2) / max(np.sum(ad ** 2), 1e-30))
    pf = [x for x in p if np.isfinite(x)]
    res[label] = dict(frames=(len(fa), len(fb)), identical_frames=sum(1 for x in p if not np.isfinite(x)),
                      psnr_mean=float(np.mean(pf)) if pf else float("inf"), psnr_min=float(min(pf)) if pf else float("inf"),
                      ssim_mean=float(np.mean(s)), ssim_min=float(min(s)),
                      audio_samples=(aa.shape[1], ab.shape[1]), audio_max_abs_diff=float(np.abs(ad).max()), audio_snr_db=float(asnr))
    pp = [min(x, 100.0) for x in p]
    axes[0].plot(pp, label=label); axes[1].plot(s, label=label)
axes[0].set_title("Per-frame PSNR (dB, capped at 100 = identical)"); axes[0].set_xlabel("frame"); axes[0].legend()
axes[1].set_title("Per-frame SSIM (luma)"); axes[1].set_xlabel("frame"); axes[1].legend(); axes[1].set_ylim(min(0.9, axes[1].get_ylim()[0]), 1.001)
fig.tight_layout(); fig.savefig(f"{out}/accuracy_curves.png", dpi=110)

# side-by-side sample frames for the first pair
label, pa, pb = pairs[0]
fa, _ = load(pa); fb, _ = load(pb)
idx = [0, len(fa) // 3, 2 * len(fa) // 3, len(fa) - 1]
fig, axes = plt.subplots(3, 4, figsize=(16, 7.5))
names = [label.split(" vs ")[0], label.split(" vs ")[1], "|diff| x10"]
for j, i in enumerate(idx):
    d = np.clip(np.abs(fa[i].astype(int) - fb[i].astype(int)) * 10, 0, 255).astype(np.uint8)
    for r, img in enumerate([fa[i], fb[i], d]):
        axes[r, j].imshow(img); axes[r, j].axis("off")
        axes[r, j].set_title(f"{names[r]} - frame {i}", fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/frames_side_by_side.png", dpi=90)
json.dump(res, open(f"{out}/accuracy.json", "w"), indent=1)
print(json.dumps(res, indent=1))
