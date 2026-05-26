import torch
import tqdm
import numpy as np
import torch.nn.functional as F

def extract_features(model, loader, device="cuda"):
    model.eval()
    features_list, pids_list, camids_list = [], [], []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Extracting Features"):
            imgs, pids, camids, _, _ = batch
            imgs = imgs.to(device)
            
            # Inference only outputs the normalized feature vector
            features = model(imgs) 
            features = F.normalize(features, p=2, dim=1)
            
            features_list.append(features.cpu())
            pids_list.append(pids)
            camids_list.append(camids)
            
    return torch.cat(features_list, dim=0), torch.cat(pids_list, dim=0), torch.cat(camids_list, dim=0)

def evaluate_reid(model, query_loader, gallery_loader, device="cuda"):
    """Computes mAP and CMC metrics for standard Re-ID benchmarks"""
    print("\n--- Starting Robust Evaluation ---")
    
    q_feats, q_pids, q_cams = extract_features(model, query_loader, device)
    g_feats, g_pids, g_cams = extract_features(model, gallery_loader, device)

    print("Computing Similarity Matrix...")
    
    sim_mat = torch.mm(q_feats, g_feats.t()).numpy()
    
    q_pids, g_pids = q_pids.numpy(), g_pids.numpy()
    q_cams, g_cams = q_cams.numpy(), g_cams.numpy()

    print("Computing Metrics...")
    all_cmc, all_AP = [], []
    
    for q_idx in range(len(q_pids)):
        q_pid = q_pids[q_idx]
        q_camid = q_cams[q_idx]

        order = np.argsort(sim_mat[q_idx])[::-1]
        matches = (g_pids[order] == q_pid).astype(np.int32)
        
        # FILTER: Remove Same ID + Same Camera (Self-matches)
        junk = (g_pids[order] == q_pid) & (g_cams[order] == q_camid)
        matches[junk] = -1
        matches = matches[matches != -1]
        
        if matches.sum() == 0: 
            continue

        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:10])

        # mAP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        all_AP.append(tmp_cmc.sum() / num_rel)

    mAP = np.mean(all_AP)
    all_cmc = np.array(all_cmc)
    rank1 = np.mean(all_cmc[:, 0])
    rank5 = np.mean(all_cmc[:, 4])
    rank10 = np.mean(all_cmc[:, 9])

    return mAP, rank1, rank5, rank10