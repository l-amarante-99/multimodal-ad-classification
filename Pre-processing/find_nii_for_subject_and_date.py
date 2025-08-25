import os
import pandas as pd
import pickle
import json

# Cache dictionary: {(ptid_path, exam_date_str): result_path}
_nii_cache = {}

def find_nii_for_subject(ptid_path, exam_date_str):
    cache_key = (ptid_path, exam_date_str)
    if cache_key in _nii_cache:
        return _nii_cache[cache_key]

    exam_prefix = exam_date_str
    preferred_folders = [
        "MPR-R__GradWarp__B1_Correction__N3",
        "MPR__GradWarp__B1_Correction__N3"
    ]

    for folder_name in preferred_folders:
        scan_path = os.path.join(ptid_path, folder_name)
        if not os.path.isdir(scan_path):
            continue

        dated_folders = [f for f in os.listdir(scan_path) if f.startswith(exam_prefix)]
        if dated_folders:
            dated_folders.sort(reverse=True)
            for folder in dated_folders:
                full_path = os.path.join(scan_path, folder)
                for root, _, files in os.walk(full_path):
                    for f in files:
                        if f.endswith(".nii") or f.endswith(".nii.gz"):
                            result = os.path.join(root, f)
                            _nii_cache[cache_key] = result
                            return result

    # Fallback to any available .nii
    nii_paths = []
    for root, _, files in os.walk(ptid_path):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                nii_paths.append(os.path.join(root, f))

    if not nii_paths:
        _nii_cache[cache_key] = None
        return None

    nii_paths.sort(key=lambda p: ("MPR-R" not in p, -os.path.getsize(p)))
    result = nii_paths[0]
    _nii_cache[cache_key] = result
    return result

def main():
    csv_path = "/home/lude14/bachelorarbeit/MRI_CNN/MRI_DIAGNOSIS.csv"
    data_dir = "/sc-projects/sc-proj-ukb-cvd/projects/theses/data/adni/MRI/ADNI"
    pkl_path = "/home/lude14/bachelorarbeit/MRI_CNN/nii_cache.pkl"
    json_path = "/home/lude14/bachelorarbeit/MRI_CNN/nii_cache.json"

    df = pd.read_csv(csv_path, usecols=["PTID", "EXAMDATE"])
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"]).dt.strftime("%Y-%m-%d")

    for _, row in df.iterrows():
        ptid_path = os.path.join(data_dir, row["PTID"])
        find_nii_for_subject(ptid_path, row["EXAMDATE"])

    print(f"Cached entries: {len(_nii_cache)}")

    # Save as pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(_nii_cache, f)
    print(f"Cache saved to: {pkl_path}")

    # Save as JSON
    json_cache = {f"{k[0]}||{k[1]}": v for k, v in _nii_cache.items()}
    with open(json_path, "w") as f:
        json.dump(json_cache, f, indent=2)
    print(f"JSON cache saved to: {json_path}")

if __name__ == "__main__":
    main()
