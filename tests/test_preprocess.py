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


if __name__ == "__main__":
    cfg = PreprocessConfig()
    processor = TextPreprocessor(cfg)
    text = """
    My experience in the coffee shop, where I felt attraction when I had the
    flu, would be called an error or misattribution in the classical view, but it’s
    no more a mistake than seeing a bee in a bunch of blobs. An influenza virus
    in my blood contributed to fever and flushing, and my brain made meaning
    from the sensations in the context of a lunch date, constructing a genuine
    feeling of attraction, in the normal way that the brain constructs any other
    mental state. If I’d had exactly the same bodily sensations while at home in
    bed with a thermometer, my brain might have constructed an instance of
    “Feeling Sick” using the same manufacturing process. (The classical view, in
    contrast, would require feelings of attraction and malaise to have different
    bodily fingerprints triggered by different brain circuitry.)
    """
    tokens = processor.process(text)
    for t in tokens:
        # 整齐地输出文本、词元和词性
        print(f"{t.text:15} {t.lemma:15} {t.pos}")