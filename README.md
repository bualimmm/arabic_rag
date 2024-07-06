# Results
The best performing model is Cohere using chain-of-thoughts prompting whose code can be found in the notebooks/azure_models_querying.ipynb and the results can be found in outputs/cleaned/cot_cleaned_cohere_results_formal.xlsx
# Experiments
**Original**: Automatically extracting text through python

**Refined**: Using Llama3-8b to refine original text

**Translation**: Translating text to English, then RAG, then translate answer to Arabic

**Cleaned**: Manually extracting text (best results)

# Labels
**-1**: context wasn't retrieved (either embedding failure or context not in the doc) but incorrectly answered (LLM failure)

**0**: context was retrieved but incorrectly answered (LLM failure)

**0.5**: context was retrieved but answered "IDK" (LLM failure)

**1**: context was retrieved and correctly answered

**2**: context wasn't retrieved (either embedding failure or context not in the doc) and correctly answered "IDK
