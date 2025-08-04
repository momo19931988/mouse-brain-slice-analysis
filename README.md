# 🧠 Automated Quantification of Fluorescence Signals in Mouse Brain Slices

This repository contains a **standardized image analysis pipeline** for the quantification of fluorescence signals in mouse brain slices.  
The workflow integrates **Python** for preprocessing, image alignment, and data extraction with **R** for statistical analysis and visualization.  

This project aims to address the challenge of performing **batch-level quantification** of brain slice fluorescence imaging, where differences in slice positioning, orientation, and brain region identification often hinder reproducible analysis.  
<img width="536" height="205" alt="image" src="https://github.com/user-attachments/assets/f55a2a0a-f6b5-4844-ab7e-9b7b78dd43c0" />

---

## 📂 Project Structure

```
project/
├── README.md         # Project introduction (this file)
├── requirements.txt  # Python dependencies
├── analysis.R        # R script for statistical analysis
├── pipeline/         # Python scripts for image analysis
│   ├── step1.py      # Channel separation
│   ├── step2.py      # Background subtraction
│   ├── step3.py      # PCA-based rotation correction
│   ├── step4.py      # Intensity normalization
│   ├── step5.py      # Coordinate extraction
│   ├── step6.py      # Region mapping
│   └── step7.py      # Batch comparison & output
├── gui/              # Graphical User Interface (Tkinter)
│   └── GUI.py
└── docs/             # Documentation and example images
    ├── workflow.png  # Example workflow diagram
    └── example.png   # Input/output illustration
```

---

## 🔹 Features
- **Channel separation** for fluorescence images (e.g., DAPI, TH, GFAP)  
- **PCA-based slice alignment** to correct orientation differences  
- **Background subtraction & intensity normalization** across groups  
- **Coordinate extraction** of fluorescence signals from individual cells or regions  
- **Standardized brain region mapping** for cross-slice comparison  
- **Batch processing** with GUI support  
- **Statistical analysis (R)** including GO/KEGG enrichment, barplots, and dotplots  
- **Export** of results as CSV tables and annotated figures  

---

## 🚀 Installation

### Python
Clone the repository and install dependencies:
```bash
git clone https://github.com/momo19931988/mouse-brain-slice-analysis.git
cd mouse-brain-slice-analysis
pip install -r requirements.txt
```

Typical dependencies include:
```txt
numpy
scipy
pandas
opencv-python
matplotlib
scikit-learn
tifffile
cellprofiler-core
tkinter        # usually built-in with Python
```

---

### R
Required R packages:
```R
install.packages(c("ggplot2", "dplyr"))
BiocManager::install(c("clusterProfiler", "org.Mm.eg.db", "enrichplot"))
```

---

## 🛠 Usage

### Run the GUI
```bash
python gui/GUI.py
```
This launches a Tkinter-based interface for **one-click batch processing**, including preprocessing, normalization, and export.

<img width="832" height="680" alt="image" src="https://github.com/user-attachments/assets/282a145f-7a84-4ea9-a212-5c23f621d048" />




### Workflow for automated alignment of mouse brain slice images.
<img width="527" height="794" alt="image" src="https://github.com/user-attachments/assets/5c192985-107b-427c-87e9-6b534691923b" />


### Downstream R analysis
```R
source("analysis.R")
```

Results include:
- CSV tables with normalized intensity values (saved in `output/`)  
- Group comparisons across brain regions  
- Figures (PNG/PDF) for statistical plots  

---

## 📊 Example Workflow
1. **Input:** Raw fluorescence images (`.tif`) from mouse brain slices  
2. **Preprocessing:** Channel separation, background correction  
3. **Alignment:** PCA-based slice rotation correction  
4. **Quantification:** Intensity normalization & coordinate extraction  
5. **Analysis:** Group-level comparison (R)  
6. **Output:** CSV files + annotated plots (example in `docs/example.png`)  

---

## 📖 Documentation
See the [`docs/`](./docs) folder for:
- Workflow diagrams (`workflow.png`)  
- Example input/output images  
- Additional technical notes  

---

## 📄 License
This project is released under the [MIT License](https://opensource.org/licenses/MIT).  

---

## ✨ Citation
Until the formal publication of the manuscript, please cite this repository as:  

This section will be updated once the related article is published.  

---
