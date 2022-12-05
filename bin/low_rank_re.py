import sys
sys.path.append("../utils/")
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import util
from matrix_completion import MatrixCompletion
from robust_pca import RobustPCA

Image = {}
Image[0] = plt.imread('../data/1.jpeg')
Image[1] = plt.imread('../data/2.jpg')
Image[2] = plt.imread('../data/3.jpg')

method_rpca = ["ALM", "APG", "SVT", "CGD"]
method_mc = ["ADMM", "SVT", "PMF", "BPMF"]
time_log = {}

for i in range(3):
    img = Image[i] / 255
    # robust pca
    img_noise = util.random_noise(img, mode='s&p', seed=2022)
    # np.save(f'../data/img_noise_{i}_rpca', img_noise)
    for meth in method_rpca:
        print(f"Robust PCA using {meth} ...")
        if meth == "ALM":
            ratio = [0.1, 0.5, 1]
            for j in range(3):
                start = time.time()
                L = {}
                S = {}
                r = ratio[j]
                for k in range(3):
                    L[k], S[k] = RobustPCA(img_noise[:,:,k], mu_ratio=r, method=meth).fit(tol=1e-7, max_iter=1000)
                img_recon = np.dstack([L[m] for m in range(3)])
                noise = np.dstack([S[m] for m in range(3)])
                # np.save(f'../data/img_recon_{i}_rpca_{meth}_r{j}', img_recon)
                # np.save(f'../data/noise_{i}_rpca_{meth}_r{j}', noise)
                end = time.time()
                time_log[f'img_{i}_rpca_{meth}_r{j}'] = end - start
        elif meth == "CGD":
            ratio = [0.0005, 0.001, 0.0015]
            for j in range(3):
                start = time.time()
                L = {}
                S = {}
                r = ratio[j]
                for k in range(3):
                    L[k], S[k] = RobustPCA(img_noise[:,:,k], mu_ratio=r, method=meth).fit(tol=1e-10, max_iter=1000)
                img_recon = np.dstack([L[m] for m in range(3)])
                noise = np.dstack([S[m] for m in range(3)])
                # np.save(f'../data/img_recon_{i}_rpca_{meth}_r{j}', img_recon)
                # np.save(f'../data/noise_{i}_rpca_{meth}_r{j}', noise)
                end = time.time()
                time_log[f'img_{i}_rpca_{meth}_r{j}'] = end - start
            # using Adam
            start = time.time()
            L = {}
            S = {}
            for k in range(3):
                L[k], S[k] = RobustPCA(img_noise[:,:,k], mu_ratio=0.01, method=meth).fit(tol=1e-10, max_iter=1000, Adam=True)
            img_recon = np.dstack([L[m] for m in range(3)])
            noise = np.dstack([S[m] for m in range(3)])
            # np.save(f'../data/img_recon_{i}_rpca_{meth}_Adam', img_recon)
            # np.save(f'../data/noise_{i}_rpca_{meth}_Adam', noise)
            end = time.time()
            time_log[f'img_{i}_rpca_{meth}_Adam'] = end - start
        elif meth == "SVT":
            delta_list = [1e-3, 2e-3, 3e-3]
            # delta_list = [0.1, 0.05, 0.01]
            for j in range(3):
                start = time.time()
                L = {}
                S = {}
                for k in range(3):
                    L[k], S[k] = RobustPCA(img_noise[:,:,k], delta=delta_list[j], mu=delta_list[j], method=meth).fit(tol=1e-7, max_iter=1000)
                    # L[k], S[k] = RobustPCA(img_noise[:,:,k], delta=delta_list[j], method=meth).fit(tol=1e-7, max_iter=1000) # old version
                img_recon = np.dstack([L[m] for m in range(3)])
                noise = np.dstack([S[m] for m in range(3)])
                # np.save(f'../data/img_recon_{i}_rpca_{meth}_r{j}', img_recon)
                # np.save(f'../data/noise_{i}_rpca_{meth}_r{j}', noise)
                end = time.time()
                time_log[f'img_{i}_rpca_{meth}_r{j}'] = end - start
        elif meth == "APG":
            start = time.time()
            L = {}
            S = {}
            for k in range(3):
                L[k], S[k] = RobustPCA(img_noise[:,:,k], method=meth).fit(tol=1e-7, max_iter=1000)
            img_recon = np.dstack([L[m] for m in range(3)])
            noise = np.dstack([S[m] for m in range(3)])
            # np.save(f'../data/img_recon_{i}_rpca_{meth}', img_recon)
            # np.save(f'../data/noise_{i}_rpca_{meth}', noise)
            end = time.time()
            time_log[f'img_{i}_rpca_{meth}'] = end - start
        else:
            print(f'Error! Method name {meth} is illegal!')

    # matrix completion
    img_noise = util.random_noise(img, mode='pepper', amount=0.5, seed=2022)
    missing_idx = (img - img_noise).astype(bool)
    # np.save(f'../data/img_noise_{i}_mc', img_noise)
    # np.save(f'../data/img_mask_{i}_mc', missing_idx)
    for j in range(3):
        for meth in method_mc:
            print(f"Matrix completion using {meth} ...")
            if meth == "ADMM":
                ratio = [0.05, 0.1, 0.15]
                for j in range(3):
                    start = time.time()
                    L = {}
                    S = {}
                    r = ratio[j]
                    for k in range(3):
                        L[k], S[k] = MatrixCompletion(img_noise[:,:,k], missing_idx[:,:,k], mu_ratio=ratio[j], method=meth).fit(tol=1e-7, max_iter=1000)
                    img_recon = np.dstack([L[m] for m in range(3)])
                    noise = np.dstack([S[m] for m in range(3)])
                    # np.save(f'../data/img_recon_{i}_mc_{meth}_r{j}', img_recon)
                    # np.save(f'../data/noise_{i}_mc_{meth}_r{j}', noise)
                    end = time.time()
                    time_log[f'img_{i}_mc_{meth}_r{j}'] = end - start
            elif meth == "SVT":
                start = time.time()
                L = {}
                S = {}
                for k in range(3):
                    L[k], S[k] = MatrixCompletion(img_noise[:,:,k], missing_idx[:,:,k], method=meth).fit(max_iter=1000)
                img_recon = np.dstack([L[m] for m in range(3)])
                noise = np.dstack([S[m] for m in range(3)])
                # np.save(f'../data/img_recon_{i}_mc_{meth}', img_recon)
                # np.save(f'../data/noise_{i}_mc_{meth}', noise)
                end = time.time()
                time_log[f'img_{i}_mc_{meth}'] = end - start
            elif meth in ["PMF", "BPMF"]:
                start = time.time()
                L = {}
                S = {}
                for k in range(3):
                    L[k], S[k] = MatrixCompletion(img_noise[:,:,k], missing_idx[:,:,k], method=meth).fit(max_iter=1000)
                img_recon = np.dstack([L[m] for m in range(3)])
                noise = np.dstack([S[m] for m in range(3)])
                # np.save(f'../data/img_recon_{i}_mc_{meth}', img_recon)
                # np.save(f'../data/noise_{i}_mc_{meth}', noise)
                end = time.time()
                time_log[f'img_{i}_mc_{meth}'] = end - start
            else:
                print(f'Error! Method name {meth} is illegal!')

# save time log
np.save('../data/time_log.npy', time_log)

