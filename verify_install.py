import sys
try:
    import langchain
    import chromadb
    import sentence_transformers
    import unstructured
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
