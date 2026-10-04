# Swahili News Classification

Research-informed multiclass classification of Swahili news articles for the ALU **NLP and Language Technologies – Formative Assignment 2**.

## Research question

**How effectively can different lexical, recurrent, convolutional, and Transformer-based approaches classify Swahili news articles, and what do their errors reveal about the strengths and limitations of sequential modelling for this dataset?**

## Dataset

The course-provided training dataset contains 23,268 records with three fields: `id`, `content`, and `category`. The six target categories are:

- `afya` — health
- `burudani` — entertainment
- `kimataifa` — international news
- `kitaifa` — national news
- `michezo` — sports
- `uchumi` — economy/business

After data-quality analysis, removal of two non-informative records, exclusion of conflicting-label texts, and same-label deduplication, the final modelling dataset contains 21,927 unique and unambiguous articles.

## Models evaluated

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Bidirectional LSTM (BiLSTM)
4. 1D CNN
5. XLM-RoBERTa (`FacebookAI/xlm-roberta-base`)

Macro-F1 is treated as the primary selection metric because the dataset is class-imbalanced.

## Final test results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8729 | 0.7623 | 0.8647 |
| Linear SVM | 0.8888 | 0.7929 | 0.8840 |
| BiLSTM | 0.8772 | 0.7978 | 0.8795 |
| 1D CNN | 0.8611 | 0.7935 | 0.8682 |
| XLM-R | **0.9198** | **0.8646** | **0.9197** |

XLM-R achieved the strongest held-out test performance. The largest improvement appears to come from multilingual contextual pretraining and subword representation rather than neural architecture alone.

## Team contribution structure

| Member | Main responsibility | Model responsibility |
|---|---|---|
| Hugues Munezero | Dataset loading, understanding, cleaning and EDA | TF-IDF + Logistic Regression |
| Mahe Digne | Text preprocessing, tokenization/representations and train/validation/test preparation | Linear SVM |
| Tapiwanashe Gift Marufu | Related-work research, model-choice justification and evaluation setup | BiLSTM and XLM-R |
| Milka Keza Isingizwe | Results comparison, evaluation, error analysis, conclusions and final integration | 1D CNN |

## Repository structure

- `notebooks/formative_2_final_polished.ipynb` — complete end-to-end notebook with EDA, preprocessing, experiments, evaluation and error analysis
- `requirements.txt` — main Python dependencies used in the project
- `.gitignore` — excludes large local data/model artifacts and temporary files

The course-provided dataset is not included in this repository. Place the training CSV in the expected notebook location before running the workflow.

## Reproducibility

The notebook sets a global random seed of `42` where supported. Neural-network results can still vary slightly across hardware, CUDA/cuDNN, TensorFlow, PyTorch and library versions.

## Main dependencies

- Python 3
- pandas
- NumPy
- matplotlib
- scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

## Academic integrity

AI tools were used as development and writing support. The submitted work remains the responsibility of the group, and each member is expected to understand and explain their assigned model, implementation choices, experimental results and limitations.
