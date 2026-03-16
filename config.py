# Default settings for dataset preparation and CSV caching.
PREPARE_DATASET_DEFAULTS = {
    "raw_data_path": "data/raw.csv",
    "cleaned_data_path": "data/cleaned.csv",
    "frac": 0.2,
    "random_state": 42,
    "num_workers": 64,
}


# Amazon Review 2023 categories processed by prepare_dataset.py.
SELECTED_CATEGORIES = [
    "Amazon_Fashion",
    "Home_and_Kitchen",
    "Books",
    "Automotive",
    "Tools_and_Home_Improvement",
    "Kindle_Store",
    "Sports_and_Outdoors",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "All_Beauty",
    "Health_and_Personal_Care",
    "Industrial_and_Scientific",
    "Grocery_and_Gourmet_Food",
    "Office_Products",
    "Electronics",
    "Pet_Supplies",
    "Arts_Crafts_and_Sewing",
    "Musical_Instruments",
    "Baby_Products",
    "Handmade_Products",
    "Software",
    "Video_Games",
    "Appliances",
    "Unknown",
    "Movies_and_TV",
]


# Default training and inference settings used by train.py.
TRAIN_DEFAULTS = {
    "seed": 42,
    "train_file": "data/cleaned.csv",
    "batch_size": 32,
    "num_workers": 16,
    "num_images_per_sample": 1,
    "image_model_name": "efficientnet_b0",
    # "text_model_name": "nreimers/MiniLM-L6-H384-uncased",
    "text_model_name": "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large",
    "hidden_dim": 512,
    "dropout": 0.1,
    "freeze_image": False,
    "freeze_text": False,
    "max_length": 256,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "patience": 1,
    "save_dir": "models",
    "save_path": "efficientnet_minilm",
    "train": True,
    "model_path": "models/efficientnet_minilm/best/best_model.pt",
    "generate_submission": False,
}
