"""Unit tests for the Porter Stemming Algorithm implementation."""

import pytest

from app.modules.retrieval.lexical.stemmer import (
    DEFAULT_TECHNICAL_EXCEPTIONS,
    PorterStemmer,
    _contains_vowel,
    _ends_cvc,
    _ends_double_consonant,
    _is_consonant,
    _measure,
    _step_1a,
    _step_1b,
    _step_1c,
    _step_2,
    _step_3,
    _step_4,
    _step_5a,
    _step_5b,
    stem,
    stem_tokens,
    stem_word,
)


class TestWordClassificationAndHelpers:
    """Tests for consonant/vowel classification, measure, and stem conditions."""

    def test_is_consonant_basic(self) -> None:
        """Standard vowels and consonants are classified correctly."""
        assert not _is_consonant("cat", 1)  # 'a' is vowel
        assert _is_consonant("cat", 0)  # 'c' is consonant
        assert _is_consonant("cat", 2)  # 't' is consonant

    def test_is_consonant_y_handling(self) -> None:
        """Letter 'y' is classified according to context."""
        # Initial y is a consonant
        assert _is_consonant("yellow", 0)
        assert _is_consonant("yes", 0)
        assert _is_consonant("y", 0)

        # y preceded by consonant is a vowel
        assert not _is_consonant("happy", 4)
        assert not _is_consonant("rhythm", 2)
        assert not _is_consonant("syzygy", 1)
        assert not _is_consonant("syzygy", 3)
        assert not _is_consonant("syzygy", 5)

        # y preceded by vowel is a consonant
        assert _is_consonant("toy", 2)
        assert _is_consonant("boy", 2)
        assert _is_consonant("say", 2)
        assert _is_consonant("play", 3)

    def test_measure_calculation(self) -> None:
        """Measure m is calculated correctly according to [C](VC)^m[V]."""
        # m = 0
        assert _measure("tr") == 0
        assert _measure("ee") == 0
        assert _measure("tree") == 0
        assert _measure("y") == 0
        assert _measure("by") == 0
        assert _measure("") == 0

        # m = 1
        assert _measure("trouble") == 1
        assert _measure("oaten") == 2  # O(V) A(V) T(C) E(V) N(C) -> 2
        assert _measure("or") == 1
        assert _measure("owns") == 1
        assert _measure("cat") == 1

        # m = 2
        assert _measure("troubles") == 2
        assert _measure("private") == 2
        assert _measure("orange") == 2

    def test_contains_vowel(self) -> None:
        """*v* condition correctly identifies presence of vowels."""
        assert _contains_vowel("motor")
        assert _contains_vowel("plaster")
        assert _contains_vowel("happy")
        assert not _contains_vowel("bl")
        assert not _contains_vowel("sk")
        assert not _contains_vowel("r")
        assert not _contains_vowel("")

    def test_ends_double_consonant(self) -> None:
        """*d condition correctly identifies double consonants."""
        assert _ends_double_consonant("hopp")
        assert _ends_double_consonant("tann")
        assert _ends_double_consonant("fall")
        assert _ends_double_consonant("hiss")
        assert _ends_double_consonant("fizz")
        assert not _ends_double_consonant("hop")
        assert not _ends_double_consonant("fast")
        assert not _ends_double_consonant("ee")  # double vowel, not consonant
        assert not _ends_double_consonant("t")
        assert not _ends_double_consonant("")

    def test_ends_cvc(self) -> None:
        """*o condition correctly identifies cvc ending where final c is not w, x, or y."""
        assert _ends_cvc("hop")
        assert _ends_cvc("wil")
        assert _ends_cvc("cav")
        assert _ends_cvc("fil")
        assert not _ends_cvc("snow")  # final 'w' excluded
        assert not _ends_cvc("box")  # final 'x' excluded
        assert not _ends_cvc("say")  # final 'y' excluded
        assert not _ends_cvc("fail")  # vowel-vowel-consonant, not cvc
        assert not _ends_cvc("to")  # length < 3
        assert not _ends_cvc("")


class TestIndividualAlgorithmSteps:
    """Isolated tests for Steps 1a through 5b."""

    def test_step_1a_plurals(self) -> None:
        """Step 1a suffix stripping."""
        assert _step_1a("caresses") == "caress"
        assert _step_1a("ponies") == "poni"
        assert _step_1a("ties") == "ti"
        assert _step_1a("caress") == "caress"
        assert _step_1a("cats") == "cat"

    def test_step_1b_participles(self) -> None:
        """Step 1b participle and progressive suffix transformations."""
        assert _step_1b("feed") == "feed"
        assert _step_1b("agreed") == "agree"
        assert _step_1b("plastered") == "plaster"
        assert _step_1b("bled") == "bled"
        assert _step_1b("motoring") == "motor"
        assert _step_1b("sing") == "sing"
        assert _step_1b("conflated") == "conflate"
        assert _step_1b("troubled") == "trouble"
        assert _step_1b("sized") == "size"
        assert _step_1b("hopping") == "hop"
        assert _step_1b("tanned") == "tan"
        assert _step_1b("falling") == "fall"
        assert _step_1b("hissing") == "hiss"
        assert _step_1b("fizzed") == "fizz"
        assert _step_1b("failing") == "fail"
        assert _step_1b("filing") == "file"

    def test_step_1c_y_to_i(self) -> None:
        """Step 1c replaces y with i if stem contains a vowel."""
        assert _step_1c("happy") == "happi"
        assert _step_1c("sky") == "sky"

    def test_step_2_relational_suffixes_including_logi(self) -> None:
        """Step 2 relational suffix replacements including logi -> log."""
        assert _step_2("geologi") == "geolog"
        assert _step_2("analogi") == "analog"
        assert _step_2("relational") == "relate"
        assert _step_2("conditional") == "condition"
        assert _step_2("rational") == "rational"
        assert _step_2("valenci") == "valence"
        assert _step_2("hesitanci") == "hesitance"
        assert _step_2("digitizer") == "digitize"
        assert _step_2("conformabli") == "conformable"
        assert _step_2("radicalli") == "radical"
        assert _step_2("differentli") == "different"
        assert _step_2("vileli") == "vile"
        assert _step_2("analogousli") == "analogous"
        assert _step_2("vietnamization") == "vietnamize"
        assert _step_2("predication") == "predicate"
        assert _step_2("operator") == "operate"
        assert _step_2("feudalism") == "feudal"
        assert _step_2("decisiveness") == "decisive"
        assert _step_2("hopefulness") == "hopeful"
        assert _step_2("callousness") == "callous"
        assert _step_2("formaliti") == "formal"
        assert _step_2("sensitiviti") == "sensitive"
        assert _step_2("sensibiliti") == "sensible"

    def test_step_3_derivational_suffixes(self) -> None:
        """Step 3 derivational suffix stripping."""
        assert _step_3("triplicate") == "triplic"
        assert _step_3("formative") == "form"
        assert _step_3("formalize") == "formal"
        assert _step_3("electriciti") == "electric"
        assert _step_3("electrical") == "electric"
        assert _step_3("hopeful") == "hope"
        assert _step_3("goodness") == "good"

    def test_step_4_suffix_removal(self) -> None:
        """Step 4 suffix removal when measure m > 1."""
        assert _step_4("revival") == "reviv"
        assert _step_4("allowance") == "allow"
        assert _step_4("inference") == "infer"
        assert _step_4("airliner") == "airlin"
        assert _step_4("gyroscopic") == "gyroscop"
        assert _step_4("adjustable") == "adjust"
        assert _step_4("defensible") == "defens"
        assert _step_4("irritant") == "irrit"
        assert _step_4("replacement") == "replac"
        assert _step_4("adjustment") == "adjust"
        assert _step_4("dependent") == "depend"
        assert _step_4("adoption") == "adopt"
        assert _step_4("homologou") == "homolog"
        assert _step_4("communism") == "commun"
        assert _step_4("activate") == "activ"
        assert _step_4("angulariti") == "angular"
        assert _step_4("homologous") == "homolog"
        assert _step_4("effective") == "effect"
        assert _step_4("bowdlerize") == "bowdler"

    def test_step_5a_and_5b_cleanup(self) -> None:
        """Step 5a and 5b cleanup transformations."""
        # 5a: probate -> probat, rate -> rate, cease -> ceas, referee -> refere, agree -> agre
        assert _step_5a("probate") == "probat"
        assert _step_5a("rate") == "rate"
        assert _step_5a("cease") == "ceas"
        assert _step_5a("referee") == "refere"
        assert _step_5a("agree") == "agre"
        # 5b: double l reduction when m > 1
        assert _step_5b("controll") == "control"
        assert _step_5b("roll") == "roll"

    def test_step_5a_regression_and_existing_cases(self) -> None:
        """Verify strict Step 5a behavior on end-to-end stem_word/stem calls."""
        assert stem_word("probate") == "probat"
        assert stem_word("rate") == "rate"
        assert stem_word("cease") == "ceas"
        assert stem_word("referee") == "refere"
        assert stem_word("agree") == "agre"


class TestCanonicalOutputsEndToEnd:
    """End-to-end tests for all canonical outputs requested by specification."""

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("relational", "relat"),
            ("geologi", "geolog"),
            ("conditional", "condit"),
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("agree", "agre"),
            ("plastered", "plaster"),
            ("motoring", "motor"),
            ("happy", "happi"),
            ("formalize", "formal"),
            ("goodness", "good"),
            ("allowance", "allow"),
            ("electriciti", "electr"),
            ("effective", "effect"),
            ("probate", "probat"),
            ("rate", "rate"),
            ("cease", "ceas"),
            ("controlling", "control"),
            ("roll", "roll"),
            ("referee", "refere"),
        ],
    )
    def test_canonical_porter_outputs(self, word: str, expected: str) -> None:
        """Verify canonical outputs through stem() and stem_word()."""
        assert stem(word) == expected
        assert stem_word(word) == expected

    def test_stem_tokens(self) -> None:
        """stem_tokens handles a sequence of words end-to-end."""
        tokens = ["caresses", "relational", "motoring", "allowance"]
        expected = ["caress", "relat", "motor", "allow"]
        assert stem_tokens(tokens) == expected


class TestGuardsAndExceptions:
    """Tests for technical terms, hyphenated tokens, digits, and apostrophes."""

    @pytest.mark.parametrize(
        "token",
        [
            "gpt-4",
            "word2vec",
            "2024",
            "3d",
            "v100",
            "k8s",
            "h100",
        ],
    )
    def test_digit_guard(self, token: str) -> None:
        """Tokens containing digits are returned unchanged."""
        assert stem(token) == token

    @pytest.mark.parametrize(
        "token",
        [
            "state-of-the-art",
            "self-attention",
            "tf-idf",
            "cross-entropy",
            "zero-shot",
            "few-shot",
        ],
    )
    def test_hyphen_guard(self, token: str) -> None:
        """Tokens containing internal hyphens are returned unchanged."""
        assert stem(token) == token

    @pytest.mark.parametrize(
        "token",
        [
            "he'd",
            "they'd",
            "she'd",
            "who'd",
            "he’d",
            "they’d",
        ],
    )
    def test_apostrophe_guard(self, token: str) -> None:
        """Tokens containing apostrophes are returned unchanged."""
        assert stem(token) == token

    @pytest.mark.parametrize(
        "token",
        [
            "",
            "a",
            "at",
            "to",
            "in",
            "is",
            "on",
        ],
    )
    def test_short_token_guard(self, token: str) -> None:
        """Tokens with length <= 2 are returned unchanged."""
        assert stem(token) == token

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("BERT", "BERT"),
            ("bert", "bert"),
            ("CNN", "CNN"),
            ("cnn", "cnn"),
            ("RNN", "RNN"),
            ("LSTM", "LSTM"),
            ("MLP", "MLP"),
            ("GAN", "GAN"),
            ("SVM", "SVM"),
            ("KNN", "KNN"),
            ("NLP", "NLP"),
            ("LLM", "LLM"),
            ("GPT", "GPT"),
            ("API", "API"),
            ("SQL", "SQL"),
            ("HTML", "HTML"),
            ("CSS", "CSS"),
            ("JSON", "JSON"),
            ("XML", "XML"),
            ("GPU", "GPU"),
            ("CPU", "CPU"),
            ("OS", "OS"),
            ("DBMS", "DBMS"),
        ],
    )
    def test_technical_exception_set_preserves_casing(
        self, token: str, expected: str
    ) -> None:
        """Technical exceptions bypass stemming and preserve original casing."""
        assert stem(token) == expected

    def test_configurable_technical_exceptions(self) -> None:
        """Custom exceptions are merged with DEFAULT_TECHNICAL_EXCEPTIONS."""
        stemmer = PorterStemmer(technical_exceptions={"rag", "mamba"})
        assert stemmer.stem("rag") == "rag"
        assert stemmer.stem("mamba") == "mamba"
        # Case-insensitive custom matching
        assert stemmer.stem("RAG") == "RAG"
        assert stemmer.stem("MAMBA") == "MAMBA"
        # Default exceptions must still work
        assert stemmer.stem("BERT") == "BERT"
        assert stemmer.stem("CNN") == "CNN"
        assert stemmer.stem("LSTM") == "LSTM"
        assert stemmer.stem("bert") == "bert"
        # Custom exceptions are in the collection
        assert "rag" in stemmer.technical_exceptions
        assert "mamba" in stemmer.technical_exceptions
        # Default exceptions are preserved in the instance
        assert "bert" in stemmer.technical_exceptions
        assert "cnn" in stemmer.technical_exceptions
        # DEFAULT_TECHNICAL_EXCEPTIONS collection remains unchanged
        assert "rag" not in DEFAULT_TECHNICAL_EXCEPTIONS
        assert "mamba" not in DEFAULT_TECHNICAL_EXCEPTIONS
        assert isinstance(DEFAULT_TECHNICAL_EXCEPTIONS, frozenset)

    def test_normal_words_lowercased_before_stemming(self) -> None:
        """Normal non-technical words are stemmed to lowercase."""
        assert stem("RELATIONAL") == "relat"
        assert stem("Caresses") == "caress"
        assert stem("MOTORING") == "motor"


class TestValidationAndImmutability:
    """Tests for input validation and output immutability."""

    def test_stem_type_error(self) -> None:
        """Passing non-string to stem() raises TypeError."""
        stemmer = PorterStemmer()
        with pytest.raises(TypeError, match="word must be a string"):
            stemmer.stem(123)  # type: ignore[arg-type]

    def test_stem_tokens_type_error(self) -> None:
        """Passing non-sequence to stem_tokens() raises TypeError."""
        stemmer = PorterStemmer()
        with pytest.raises(TypeError, match="tokens must be a Sequence"):
            stemmer.stem_tokens(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="tokens must be a Sequence"):
            stemmer.stem_tokens("string_is_not_sequence_of_tokens")  # type: ignore[arg-type]

    def test_stem_tokens_item_type_error(self) -> None:
        """Passing sequence containing non-string items raises TypeError."""
        stemmer = PorterStemmer()
        with pytest.raises(TypeError, match="word must be a string"):
            stemmer.stem_tokens(["cat", 123])  # type: ignore[list-item]

    def test_technical_exceptions_type_error(self) -> None:
        """Passing invalid technical_exceptions raises TypeError."""
        with pytest.raises(TypeError, match="technical_exceptions must be an iterable"):
            PorterStemmer(technical_exceptions=123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="technical_exceptions must be an iterable"):
            PorterStemmer(technical_exceptions="string")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="technical_exception item must be a string"):
            PorterStemmer(technical_exceptions=["valid", 123])  # type: ignore[list-item]

    def test_immutability_and_fresh_list(self) -> None:
        """stem_tokens returns a new list and does not mutate the input list."""
        input_list = ["caresses", "ponies", "ties"]
        result = stem_tokens(input_list)
        assert result == ["caress", "poni", "ti"]
        assert result is not input_list
        assert input_list == ["caresses", "ponies", "ties"]

    def test_technical_exceptions_property_returns_frozenset(self) -> None:
        """technical_exceptions property returns an immutable frozenset."""
        stemmer = PorterStemmer()
        exceptions = stemmer.technical_exceptions
        assert isinstance(exceptions, frozenset)
        assert "bert" in exceptions
        assert "cnn" in exceptions
