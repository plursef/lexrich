from lexrich.config import PreprocessConfig
from lexrich.preprocess import TextPreprocessor


def test_preprocess_filters_stopwords():
    cfg = PreprocessConfig()
    processor = TextPreprocessor(cfg)
    tokens = processor.process("The happy cats are running joyfully in the garden")
    lemmas = [t.lemma for t in tokens]
    assert "happy" in lemmas
    assert "the" not in lemmas
    assert all(len(t.lemma) >= cfg.min_token_len for t in tokens)
