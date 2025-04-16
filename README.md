# DeepFace Testing Scratchpad for Graduation Project

This repository contains a Jupyter Notebook (`Report.ipynb`) used for experimenting with the [DeepFace](https://github.com/serengil/deepface) library for facial recognition tasks. It serves as a **testing ground and scratchpad** for functionalities being explored for my graduation project, which involves identifying individuals and classifying them based on predefined lists (e.g., "passed" or "banned").

**Note:** This notebook represents experimental code and work in progress. It's designed for rapid testing and exploration of DeepFace capabilities, particularly its `find` function and face extraction features.

## Core Functionality Explored

The `Report.ipynb` notebook demonstrates the following:

1.  **Database Setup:** Defines a simple directory structure (`DB/`) where subdirectories represent individual identities (e.g., `DB/faris/`, `DB/rasheed/`) and contain corresponding face images.
2.  **User Classification Lists:** Defines Python lists (`PASSED_USERS`, `BANNED_USERS`) to specify which identities belong to allowed or restricted groups.
3.  **Face Detection & Extraction (Preprocessing):** Implements a crucial preprocessing step (`extract_and_save_faces` function) using `DeepFace.extract_faces`. This detects all faces within a given input image (even complex ones with multiple people or unclear faces) and saves each detected face as a separate temporary image file. This allows for individual analysis of each face.
4.  **Face Recognition:** Uses the `DeepFace.find` function within `check_user_status` to compare each *extracted* face image against the established database (`DB`).
5.  **Status Determination:** Based on the best match found by `DeepFace.find` for an extracted face, the code checks if the matched identity is present in the `PASSED_USERS` or `BANNED_USERS` lists and assigns a status (e.g., `PASSED`, `BANNED`, `UNKNOWN_USER`, `NO_MATCH`).
6.  **Workflow Orchestration:** The `process_image_and_check_status` function ties the extraction and checking steps together, managing temporary files and providing results for all faces found in the original input image.
7.  **(Optional) Directory Processing:** Includes a commented-out function (`process_images_in_directory`) to potentially process all images within a specified test directory automatically.

## How It Works

1.  **Database:** Create a main folder (e.g., `DB`). Inside it, create subfolders named after each person's unique ID (e.g., `faris`, `rasheed`). Place one or more clear images of *only that person* inside their respective folder. DeepFace will automatically create a cache file (e.g., `representations_vgg-face.pkl`) in the `DB` folder the first time it analyzes it. **Delete this `.pkl` file if you add, remove, or change images in the database.**
2.  **Preprocessing (Handling Complex Images):** When an input image is provided to `process_image_and_check_status`:
    *   `extract_and_save_faces` is called.
    *   It uses a face detector (default: `retinaface`) to find all faces in the input image.
    *   Each detected face is cropped and saved as a separate temporary `.jpg` file.
3.  **Recognition & Classification:**
    *   The code iterates through each temporary face file created in the previous step.
    *   `check_user_status` is called for each temporary face file.
    *   `DeepFace.find` searches the `DB` for the closest match to the current temporary face.
    *   If a match is found above the internal threshold, the identity (folder name) of the best match is extracted.
    *   This identity is compared against the `PASSED_USERS` and `BANNED_USERS` lists to determine the status for that specific face.
4.  **Cleanup:** The temporary directory and all extracted face images are automatically deleted after processing.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Create a Python environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate the environment
    # Windows:
    .\.venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create the Database:** Create a folder named `DB` (or update `DATABASE_PATH` in the notebook) and populate it with identity subfolders and images as described in "How It Works".

## Usage

1.  Launch Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
2.  Open the `scratch.ipynb` notebook.
3.  Make sure the `DATABASE_PATH`, `PASSED_USERS`, and `BANNED_USERS` variables are correctly set for your database structure.
4.  Run the cells sequentially to define the functions.
5.  To test a single complex image:
    *   Update the `TEST_IMAGE_COMPLEX` variable in the final execution cell with the path to your test image.
    *   Run the final cell. The output will show the processing steps and the final status determined for each face found in the image.
6.  To test all images in a directory:
    *   Uncomment the last cell block (containing the call to `process_images_in_directory`).
    *   Set the `TEST_IMAGES_DIRECTORY` variable to the path of your folder containing test images.
    *   Run the cell.

## Project Context

This notebook is a development tool and exploration space for my graduation project. The goal is to refine a system that can accurately identify individuals from images or video streams and classify them based on predefined access lists, handling real-world challenges like multiple people in frame or non-ideal face captures. The preprocessing step of extracting individual faces is a key part of this strategy.

