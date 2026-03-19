try:
    import langchain
    print(f"langchain version: {langchain.__version__}")
except ImportError:
    print("langchain not found")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("langchain_text_splitters found")
except ImportError:
    print("langchain_text_splitters not found")

try:
    import langchain_community
    print("langchain_community found")
except ImportError:
    print("langchain_community not found")
