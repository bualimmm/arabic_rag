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
