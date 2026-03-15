import argparse
import os
import re
import json
import math
import pandas as pd
from datasets import load_dataset

from config import PREPARE_DATASET_DEFAULTS, SELECTED_CATEGORIES


def normalize_metadata(dp):
    out = {}

    out["main_category"] = dp.get("main_category", "")
    out["title"] = dp.get("title", "")
    out["average_rating"] = dp.get("average_rating", None)
    out["rating_number"] = dp.get("rating_number", None)

    # features: list -> string
    feats = dp.get("features") or []
    out["features"] = " | ".join([f.replace('|', ' ').strip() for f in feats if isinstance(f, str)])

    # description: list -> string
    desc = dp.get("description") or []
    out["description"] = " ".join([d.strip() for d in desc if isinstance(d, str)])

    # images: keep hi_res and large only
    images = dp.get("images") or {}
    hi_res = images.get("hi_res", []) if isinstance(images, dict) else []
    hi_res = [url for url in hi_res if url is not None and url]
    out["images_hi_res"] = " | ".join(hi_res)

    large = images.get("large", []) if isinstance(images, dict) else []
    large = [url for url in large if url is not None and url]
    out["images_large"] = " | ".join(large)

    # videos: keep URLs only
    videos = dp.get("videos") or {}
    urls = videos.get("url", []) if isinstance(videos, dict) else []
    urls = [url for url in urls if url is not None and url]
    out["video_urls"] = " | ".join(urls)

    details = dp.get("details") or ""
    details = format_json_details(details)
    out["details"] = details

    for key in ["main_category", "title", "features", "description", "details"]:
        if out[key] is not None:
            out[key] = out[key] \
                .replace('\t', ' ') \
                .replace('\n', ' ') \
                .replace('\r', '') \
                .strip()

    return out

def filter_metadata(dp):
    def is_missing(x):
        if x is None:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        if isinstance(x, str) and not x.strip():
            return True
        if isinstance(x, list) and len(x) == 0:
            return True
        return False

    title = dp.get("title")
    features = dp.get("features")
    description = dp.get("description")
    main_category = dp.get("main_category")
    average_rating = dp.get("average_rating")
    rating_number = dp.get("rating_number")
    images_hi_res = dp.get("images_hi_res")
    images_large = dp.get("images_large")

    required_fields = [main_category, title, features, description, average_rating, rating_number]
    if any(is_missing(x) for x in required_fields):
        return False

    cleaned_metadata = f"{title} {features} {description}"
    if len(cleaned_metadata.strip()) <= 30:
        return False

    if is_missing(images_hi_res) and is_missing(images_large):
        return False

    return True

def sample_category(group, min_samples=10000, base_frac=0.01, random_state=42):
    n = len(group)
    target_n = max(int(n * base_frac), min_samples)
    target_n = min(target_n, n)  # cannot sample more than available
    return group.sample(n=target_n, random_state=random_state)

def load_raw_data(save_path="data/raw.csv", frac=0.01, random_state=42):
    if os.path.exists(save_path):
        print(f"Loading existing raw data from {save_path}")
        df = pd.read_csv(save_path, keep_default_na=False)
        print("Loaded:", df.shape)
        return df

    all_rows = []
    for idx, category in enumerate(SELECTED_CATEGORIES):
        print(f"Processing category {idx+1}/{len(SELECTED_CATEGORIES)}: {category}")
        
        # Load metadata dataset
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            cache_dir="./hf_datasets"
        )

        # Keep only specified fields
        fields_to_keep = ['main_category', 'title', 'average_rating', 'features', 'description', 'images', 'videos', 'details', 'rating_number']
        meta_dataset = meta_dataset.select_columns(fields_to_keep)

        normalized_meta_dataset = meta_dataset.map(
            normalize_metadata,
            num_proc=PREPARE_DATASET_DEFAULTS["num_workers"]
        )

        filtered_meta_dataset = normalized_meta_dataset.filter(
            filter_metadata,
            num_proc=PREPARE_DATASET_DEFAULTS["num_workers"]
        )

        temp_df = filtered_meta_dataset.to_pandas()

        # temp_df = temp_df.groupby("main_category", group_keys=False).apply(
        #     lambda x: x.sample(frac=frac, random_state=random_state)
        # ).reset_index(drop=True)

        temp_df["main_category"] = category
        if frac is not None:
            temp_df = sample_category(
                temp_df,
                min_samples=10000,
                base_frac=frac,
                random_state=random_state
            ).reset_index(drop=True)

        all_rows.extend(temp_df.to_dict('records'))

        del meta_dataset
        del normalized_meta_dataset
        del filtered_meta_dataset
        del temp_df

    df = pd.DataFrame(all_rows)
    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8"
    )

    print("Saved:", df.shape)
    print(f"Total categories processed: {len(SELECTED_CATEGORIES)}")
    
    return df

def preprocess_data(file_path, save_path="data/cleaned.csv"):
    df = pd.read_csv(file_path, keep_default_na=False)

    # Ensure string columns are strings
    df["main_category"] = df["main_category"].astype(str)
    df["title"] = df["title"].astype(str)
    df["features"] = df["features"].astype(str)
    df["description"] = df["description"].astype(str)
    df["details"] = df["details"].astype(str)
    df["images_hi_res"] = df["images_hi_res"].astype(str)
    df["images_large"] = df["images_large"].astype(str)
    df["video_urls"] = df["video_urls"].astype(str)
    df["average_rating"] = df["average_rating"].astype(float)
    df["rating_number"] = df["rating_number"].astype(float)

    # Save processed data
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Processed data saved to {save_path}")

    # Save images
    # import requests
    # img_path = "data/images"
    # os.makedirs(img_path, exist_ok=True)
    # img_filepath_list = []
    # for idx, row in df.iterrows():
    #     img_filename = ""
    #     for img_col in ["images_hi_res", "images_large"]:
    #         img_urls = row[img_col].split(" | ") if row[img_col] else []
    #         for i, url in enumerate(img_urls):
    #             if url:
    #                 img_filename = f"{img_col}_{idx}_{i}.jpg"
    #                 img_filepath = os.path.join(img_path, img_filename)
    #                 # Here you would add code to download the image from the URL and save it to img_filepath
    #                 # For example, using requests:
    #                 try:
    #                     response = requests.get(url, timeout=5)
    #                     if response.status_code == 200:
    #                         with open(img_filepath, 'wb') as f:
    #                             f.write(response.content)
    #                 except Exception as e:
    #                     print(f"Failed to download {url}: {e}")

    #                 break  # Only download the first image from each column

    #         if img_urls:
    #             break  # Only process the first non-empty image column

    #     img_filepath_list.append(img_filename)
    
    # df['image_filename'] = img_filepath_list
    # df.to_csv(save_path, index=False, encoding="utf-8")

    return df

def analyse_data(df):
    print(df.info())

    print("\nItems per category:")
    category_counts = df["main_category"].value_counts()
    category_pct = df["main_category"].value_counts(normalize=True) * 100
    category_df = pd.DataFrame({
        'Count': category_counts,
        'Percentage': category_pct
    })
    print(category_df.to_string())

    print("\nRating Distribution:")
    rating_counts = df["average_rating"].value_counts()
    rating_pct = df["average_rating"].value_counts(normalize=True) * 100
    rating_df = pd.DataFrame({
        'Count': rating_counts,
        'Percentage': rating_pct
    })
    print(rating_df.to_string())

    print("\nText length statistics:")
    df["text"] = df.apply(create_text, axis=1)
    text_lengths = df["text"].apply(len)
    print(f"Text length - mean: {text_lengths.mean():.2f}, min: {text_lengths.min()}, max: {text_lengths.max()}")
    print(df.iloc[0]["text"])

    print("\nFirst row sample:")
    print(df.iloc[0].T)

def format_json_details(details_str):
    try:
        details = json.loads(details_str)  # Convert string to dict
        if isinstance(details, dict):
            formatted = []
            for key, value in details.items():
                formatted.append(f"{key}: {value}".replace(';', ' ').strip())
            return " | ".join(formatted)
        else:
            return str(details_str)  # If it's not a dict, return as is
    except Exception as e:
        # If parsing fails, return the original string
        # print(f"Warning: Failed to parse details: {e}")
        return str(details_str)

def clean_text(text):
    # remove escaped quotes
    text = text.replace("\\'", "").replace('\\"', "")

    # remove URLs
    text = re.sub(r'(https?://\S+|www\.\S+)', ' ', text, flags=re.IGNORECASE)

    # remove html tags
    text = re.sub(r'<.*?>', ' ', text)

    # remove hashtags (#something)
    # text = re.sub(r'#\w+', ' ', text)
    
    # remove dates (2024-01-01, 01/01/2024)
    # text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", text)
    # text = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", " ", text)
    
    # remove time formats (12:30, 09:15:20)
    # text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', ' ', text)
    
    # remove repeated punctuation (!!! -> ! , ??? -> ?)
    # text = re.sub(r'([!?.,])\1+', r'\1', text)
    
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def filter_valid_entries(df, save=False, save_path="data/filtered.csv"):
    old_df = df.copy()

    # Keep rows where title is non-null and non-empty
    df = df[df["title"].apply(lambda x: pd.notna(x) and str(x).strip() != "")]

    # Keep rows where average_rating is non-null and non-empty
    df = df[df["average_rating"].apply(lambda x: pd.notna(x) and str(x).strip() != "" and x is not None)]

    text_cols = ["title", "features", "description", "details"]
    pre_clean = df[text_cols].copy()

    for col in text_cols:
        df[col] = df[col].apply(clean_text)

    changed_mask = (df[text_cols] != pre_clean).any(axis=1)
    affected_rows = df[changed_mask]

    dropped = len(old_df) - len(df)
    print(f"filter_valid_entries: kept {len(df)}/{len(old_df)} rows ({dropped} dropped).")
    print(f"filter_valid_entries: {len(affected_rows)} rows had text modified by clean_text.")
    if not affected_rows.empty:
        sample_idx = affected_rows.index[0]
        changed_cols = [c for c in text_cols if df.loc[sample_idx, c] != pre_clean.loc[sample_idx, c]]
        print(f"Sample affected row (index {sample_idx}), changed columns: {changed_cols}")
        for c in changed_cols:
            print(f"  [{c}] before: {pre_clean.loc[sample_idx, c]!r}")
            print(f"  [{c}]  after: {df.loc[sample_idx, c]!r}")

    if save:
        df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"Saved filtered data to {save_path}")

    return df.reset_index(drop=True)

def create_text(row):
    product_info_list = []
    if pd.notna(row['main_category']) and str(row['main_category']).strip():
        product_info_list.append(f"Main category: {row['main_category']}")
    if pd.notna(row['title']) and str(row['title']).strip():
        product_info_list.append(f"Title: {row['title']}")
    if pd.notna(row['features']) and str(row['features']).strip():
        product_info_list.append(f"Features: {row['features']}")
    if pd.notna(row['description']) and str(row['description']).strip():
        product_info_list.append(f"Description: {row['description']}")
    if pd.notna(row['details']) and str(row['details']).strip():
        product_info_list.append(f"Details: {row['details']}")
    
    product_info_string = "\n".join(product_info_list)

    return product_info_string

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Amazon review metadata datasets.")
    parser.add_argument(
        "--raw-data-path",
        default=PREPARE_DATASET_DEFAULTS["raw_data_path"],
        help="Path to the cached raw CSV produced from Hugging Face metadata.",
    )
    parser.add_argument(
        "--cleaned-data-path",
        default=PREPARE_DATASET_DEFAULTS["cleaned_data_path"],
        help="Path to the cleaned CSV used for downstream training.",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=PREPARE_DATASET_DEFAULTS["frac"],
        help="Sampling fraction per category when building the raw dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=PREPARE_DATASET_DEFAULTS["random_state"],
        help="Random seed used for sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs("data", exist_ok=True)
    raw_dir = os.path.dirname(args.raw_data_path)
    cleaned_dir = os.path.dirname(args.cleaned_data_path)
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
    if cleaned_dir:
        os.makedirs(cleaned_dir, exist_ok=True)

    if os.path.exists(args.cleaned_data_path):
        print(f"Loading existing cleaned data from {args.cleaned_data_path}")
        df = pd.read_csv(args.cleaned_data_path, keep_default_na=False)
    else:
        df = load_raw_data(
            args.raw_data_path,
            frac=args.frac,
            random_state=args.random_state,
        )
        df = preprocess_data(args.raw_data_path, save_path=args.cleaned_data_path)

    analyse_data(df)


if __name__ == "__main__":
    main()
