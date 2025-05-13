import argparse
import json
import os

from datasets import Dataset, Features, Image, Value, load_from_disk
from PIL import Image as PILImage


def convert_json_to_dataset(args):
    """Convert a JSON file containing image metadata and GPT-4 outputs to a Hugging Face Dataset object.

    This function processes a JSON file where each entry contains image path information and
    associated GPT-4 generated text. It loads the images, extracts the corresponding text,
    and creates a dataset with 'image', 'text_data', and 'answer' fields.

    Args:
        args: An argument object with the following attributes:
            - json_file (str): Path to the JSON file containing the dataset information
            - image_base_dir (str): Base directory where images are stored
            - output_type (str): The type of GPT-4 output to use (defaults to 'summarized')
            - output_file (str, optional): Path to save the created dataset

    Returns:
        Dataset: A Hugging Face Dataset object containing the processed data
    """
    # Load JSON data
    with open(args.json_file, "r") as f:
        data = json.load(f)

    # Initialize lists to store data
    image_paths = []
    text_prompts = []
    answers = []

    # Process each item in the JSON
    for item in data:
        index = item["index"]
        base_dir = item["source"]["base_dir"]
        image_path = os.path.join(args.image_base_dir, base_dir, index)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}, skipping...")
            continue

        # Load the image as PIL Image object
        try:
            image = PILImage.open(image_path)
            # Convert to RGB if needed (handles PNG with transparency)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}, skipping...")
            continue

        # Get GPT-4 output (using 'summarized' type as default, can be modified)
        answer = item["gpt4_output"][args.output_type]

        # Add data to lists
        image_paths.append(image)  # Store PIL image object directly
        text_prompts.append("Describe the image")
        answers.append(answer)

    # Create the dataset
    dataset_dict = {"image": image_paths, "text_data": text_prompts, "answer": answers}

    # Create Features specification
    features = Features(
        {"image": Image(), "text_data": Value("string"), "answer": Value("string")}
    )

    # Create the dataset
    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Save the dataset if output file is specified
    if args.output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the dataset to disk
        print(f"Saving dataset to {args.output_file}")
        dataset.save_to_disk(args.output_file)
    else:
        print("No output file specified, dataset not saved")

    return dataset


if __name__ == "__main__":
    """Convert json file to Hugging Face Dataset.""" ""
    # Example usage:
    # python convert_json_to_dataset.py --json_file ~/scratch/gpt4o_output_dataset_v1.json
    # --image_base_dir ~/scratch/dataset_v1/ --output_file json_dataset
    parser = argparse.ArgumentParser(
        description="Convert JSON file to Hugging Face Dataset"
    )
    parser.add_argument(
        "--json_file", type=str, required=True, help="Path to JSON file"
    )
    parser.add_argument(
        "--image_base_dir", type=str, required=True, help="Base directory for images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output path for dataset. If file exists, it wont be overwritten",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="summarized",
        choices=["structured", "short", "summarized"],
        help="Which GPT-4 output to use",
    )

    args = parser.parse_args()
    assert os.path.exists(args.json_file), f"JSON file {args.json_file} does not exist"
    assert os.path.exists(
        args.image_base_dir
    ), f"Image base directory {args.image_base_dir} does not exist"
    if os.path.exists(args.output_file):
        # Load the dataset
        dataset = load_from_disk(args.output_file)
        # Now you can use the dataset
        print(f"Dataset has {len(dataset)} examples")
        print(dataset.column_names)
        print(dataset[0])
    else:
        convert_json_to_dataset(args)
