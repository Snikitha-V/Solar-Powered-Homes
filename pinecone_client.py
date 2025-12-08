"""legacy shim, no longer needed."""
    raise

# Re-export public names
for _name in dir(_pinecone):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_pinecone, _name)
