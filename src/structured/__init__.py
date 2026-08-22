"""Generic structured-output extraction: text + JSON Schema -> validated dict.

Pipeline (see ARCHITECTURE notes in each module):

    input text + JSON Schema file
        -> schema.load_and_validate_schema()   (meta-validate + subset check)
        -> model_builder.build_model()          (JSON Schema -> Pydantic model)
        -> extractor.extract()                  (prompt -> parse -> validate -> retry)
        -> plain dict, ready to json.dumps()

Pydantic is an implementation detail of this package only: nothing outside
`src/structured/` should need to import it. Callers (the CLI) work with
plain dicts and the shared `GatewayError` hierarchy from `src.core.errors`.
"""

from .extractor import ExtractionResult, extract
from .schema import load_and_validate_schema

__all__ = ["ExtractionResult", "extract", "load_and_validate_schema"]
