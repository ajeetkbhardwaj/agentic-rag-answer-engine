from ingestion.chunking import DocumentChunker


def test_chunk_text_basic():
    text = "This is a sentence. " * 200
    chunks = DocumentChunker.chunk_text(text, chunk_size=200, chunk_overlap=20, document_id='doc1', source='test')
    assert len(chunks) > 0
    assert all(hasattr(c, 'content') for c in chunks)


def test_chunk_by_sections():
    text = "Section one.\n\nSection two.\n\nSection three."
    sections = DocumentChunker.chunk_by_sections(text, document_id='doc2', source='test')
    assert len(sections) == 3
