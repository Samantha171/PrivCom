import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import log10

# ---------- Setup & Utilities ----------
def ensure_dirs():
    os.makedirs("outputs/pmc", exist_ok=True)
    os.makedirs("outputs/roi", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

def psnr_mse(original, reconstructed):
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    if mse == 0:
        return float('inf'), 0.0
    max_i = np.max(original)
    psnr = 10 * log10((max_i ** 2) / mse)
    return psnr, mse

def file_size_kb(path):
    return os.path.getsize(path) / 1024.0

def save_img(path, arr):
    cv2.imwrite(path, (np.clip(arr, 0, 1) * 255).astype(np.uint8))

def plot_compare(original, processed, title, outpath):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title("Processed")
    plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------- Algorithm 1: PMC ----------
def pmc_compress(img_path, out_path, k=50, noise_std=0.02):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    arr = img.astype(np.float32) / 255.0

    U, S, Vt = np.linalg.svd(arr, full_matrices=False)
    U_k, S_k, Vt_k = U[:, :k], np.diag(S[:k]), Vt[:k, :]
    reconstructed = np.dot(U_k, np.dot(S_k, Vt_k))

    noise = np.random.normal(0, noise_std, reconstructed.shape)
    private_img = np.clip(reconstructed + noise, 0, 1)
    save_img(out_path, private_img)
    return arr, private_img

# ---------- Algorithm 2: ROI ----------
def roi_mask_and_compress(img_path, out_path, jpeg_quality=40):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        # fallback: blur center region if no face detected
        cx, cy = 128, 128
        img[cy-50:cy+50, cx-50:cx+50] = cv2.GaussianBlur(img[cy-50:cy+50, cx-50:cx+50], (99, 99), 30)
    else:
        for (x, y, w, h) in faces:
            img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (99, 99), 30)

    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    gray_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray, gray_processed

# ---------- Main Comparison ----------
def run_ppcp():
    ensure_dirs()
    images = ["images/dataset1.jpg", "images/dataset2.jpg"]
    results = []

    for idx, img_path in enumerate(images, start=1):
        print(f"\nProcessing Image {idx}: {img_path}")

        # ---------- PMC ----------
        pmc_out = f"outputs/pmc/img{idx}_pmc.jpg"
        orig_gray, pmc_gray = pmc_compress(img_path, pmc_out, k=50, noise_std=0.02)
        pmc_psnr, pmc_mse = psnr_mse(orig_gray, pmc_gray)
        pmc_size = file_size_kb(pmc_out)

        # ---------- ROI ----------
        roi_out = f"outputs/roi/img{idx}_roi.jpg"
        orig_gray2, roi_gray = roi_mask_and_compress(img_path, roi_out)
        roi_psnr, roi_mse = psnr_mse(orig_gray2 / 255.0, roi_gray)
        roi_size = file_size_kb(roi_out)

        orig_size = file_size_kb(img_path)
        pmc_ratio = 100 * (1 - pmc_size / orig_size)
        roi_ratio = 100 * (1 - roi_size / orig_size)

        # ---------- Print Metrics ----------
        print(f"Original: {orig_size:.2f} KB")
        print(f"PMC -> {pmc_size:.2f} KB | Reduction: {pmc_ratio:.2f}% | PSNR: {pmc_psnr:.2f} dB | MSE: {pmc_mse:.5f}")
        print(f"ROI -> {roi_size:.2f} KB | Reduction: {roi_ratio:.2f}% | PSNR: {roi_psnr:.2f} dB | MSE: {roi_mse:.5f}")

        # ---------- Individual Comparison Plots ----------
        plot_compare(orig_gray, pmc_gray, f"PMC - Image {idx}", f"plots/pmc_img{idx}.png")
        plot_compare(orig_gray2, roi_gray * 255.0, f"ROI - Image {idx}", f"plots/roi_img{idx}.png")

        # ---------- Side-by-Side PMC vs ROI ----------
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(orig_gray, cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pmc_gray, cmap='gray')
        plt.title("PMC (SVD + Noise)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(roi_gray, cmap='gray')
        plt.title("ROI (Masked + JPEG)")
        plt.axis("off")

        plt.suptitle(f"Algorithm Comparison - Image {idx}")
        plt.tight_layout()
        plt.savefig(f"plots/compare_img{idx}.png")
        plt.close()
        print(f"Saved combined comparison plot: plots/compare_img{idx}.png")

        better = "PMC" if pmc_psnr > roi_psnr else "ROI"
        results.append({
            "image": os.path.basename(img_path),
            "better": better,
            "pmc_psnr": pmc_psnr,
            "roi_psnr": roi_psnr,
            "pmc_ratio": pmc_ratio,
            "roi_ratio": roi_ratio
        })

    # ---------- Summary ----------
    print("\n====== Summary ======")
    for r in results:
        print(f"{r['image']}: {r['better']} gives better quality "
              f"(PMC PSNR={r['pmc_psnr']:.2f} dB vs ROI PSNR={r['roi_psnr']:.2f} dB)")
        print(f"PMC reduction: {r['pmc_ratio']:.2f}% | ROI reduction: {r['roi_ratio']:.2f}%")

    print("\nAll outputs saved under 'outputs/' and comparison plots under 'plots/'")


# ---------- Run ----------
if __name__ == "__main__":
    run_ppcp()
