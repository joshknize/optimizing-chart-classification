# Optimizing Chart Classification

This repository contains code for reproducing the results of "Optimizing Chart Classification: A Study of Data Augmentation and Training Strategies". You can reproduce our state-of-the-art results (94.90% macro-averaged F1 score on the 2024 CHART-Info test dataset) using `config/config.json`, or you can edit the file to run your own experiments. 

## Project Structure

- `python/` - Main source code
  - `main.py` - Entry point for running experiments
  - `occ/` - Core modules
- `config/` - Configuration files
- `output/` - Output directory for logs, CSV results and statistics, error analysis, and plots

## How To

1. Install required Python packages using `requirements.txt`.

2. Use `data_split.json`, along with a copy of the 2024 CHART-Info training dataset, to replicate our custom split of the data. This is an important step, because our split of the data ensures that charts from the same scientific papers do not appear across both the training and validation split, which can lead to overfitting. 

3. Edit and save your config file (`/config`) to set paths and parameters.

4. Run training from the command line. For example:
   ```bash
   python python/main.py config/my_config.json
   ```

5. Find logs, confusion matrices, summary CSV data, and error analysis plots in `output/` (as specified in config file).

---

For more details, see comments in the code and config file, as well as the paper.
