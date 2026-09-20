"""Build-time MLIR highlighting, with no browser scripts or network."""
from pygments.lexer import RegexLexer, words
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Text


class MLIRLexer(RegexLexer):
    name = "Tiga MLIR"
    aliases = ["mlir"]
    filenames = ["*.mlir"]
    tokens = {"root": [
        (r"//[^\n]*", Comment.Single),
        (r"\s+", Text.Whitespace),
        (r'"[A-Za-z_][\w]*\.[\w.]+"(?=\s*\()', Name.Function),
        (r'"(?:[^"\\]|\\.)*"', String),
        (r"%[\w.$-]+(?:#[0-9]+)?", Name.Variable),
        (r"@[\w.$-]+", Name.Function),
        (r"\^[\w.$-]+", Name.Label),
        (r"![\w.]+", Keyword.Type),
        (r"#[\w.]+", Name.Constant),
        (r"\b(?:f(?:16|32|64)|bf16|[su]?i[0-9]+|index)\b", Keyword.Type),
        (words(("tensor", "memref", "vector", "array"), suffix=r"\b"), Keyword.Type),
        (words(("module", "return", "func", "true", "false", "loc"), suffix=r"\b"), Keyword),
        (r"\b[A-Za-z_][\w]*\.[\w.]+", Name.Function),
        (r"[-+]?(?:0x[\da-fA-F]+|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)", Number),
        (r"->|[=+*?]", Operator),
        (r"[{}()\[\]<>,:;]", Punctuation),
        (r"[A-Za-z_][\w-]*", Name.Attribute),
        (r".", Text),
    ]}


def on_startup(**kwargs):
    # Populate both tables: SuperFences and inlinehilite use alias lookup.
    from pygments import lexers
    lexers.LEXERS["TigaMLIRLexer"] = (
        __name__, MLIRLexer.name, tuple(MLIRLexer.aliases), tuple(MLIRLexer.filenames), (),
    )
    lexers._lexer_cache[MLIRLexer.name] = MLIRLexer
