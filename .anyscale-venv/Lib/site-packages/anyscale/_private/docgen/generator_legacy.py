import re
from typing import Dict, List, Tuple


ANYSCALE_SDK_INTRO = """\
The AnyscaleSDK class must be constructed in order to make calls to the SDK. This class allows you to create an authenticated client in which to use the SDK.

| Param | Type | Description |
| :--- | :--- | :--- |
| `auth_token` | Optional String | Authentication token used to verify you have permissions to access Anyscale. If not provided, permissions default to the credentials set for your current user. Credentials can be set by following the instructions on this page: https://console.anyscale.com/credentials |

**Example**
```python
from anyscale import AnyscaleSDK

sdk = AnyscaleSDK()
```
"""


def _build_model_to_module_mapping() -> Dict[str, str]:
    """Build mapping from model name (lowercase) to module filename.

    This dynamically discovers the mapping from ALL_MODULES configuration,
    eliminating the need for hardcoded constants.
    """
    # Import here to avoid circular imports
    from anyscale._private.docgen.__main__ import (  # noqa: PLC0415 - codex_reason("gpt5.2", "avoid circular import at module load")
        ALL_MODULES,
    )

    model_name_to_file = {}

    for module in ALL_MODULES:
        # Current models (Python types)
        for model_type in module.models:
            model_name_to_file[model_type.__name__.lower()] = module.filename

        # Legacy models (strings)
        if module.legacy_sdk_models:
            for model_name in module.legacy_sdk_models:
                model_name_to_file[model_name.lower()] = module.filename

    return model_name_to_file


def _transform_legacy_links(text: str, for_legacy_file: bool = True) -> str:
    """Transform legacy model links to include proper cross-module references.

    Args:
        text: The text containing links to transform
        for_legacy_file: If True, generate links for legacy files (in legacy/ subfolder).
                        If False, generate links for main files with -legacy anchors.
    """
    # Build the mapping dynamically
    model_mapping = _build_model_to_module_mapping()

    def replace_link(match):
        model_name = match.group(1)
        if model_name in model_mapping:
            module_file = model_mapping[model_name]
            if for_legacy_file:
                # In legacy files, link to other legacy files without -legacy suffix
                return f"({module_file}#{model_name})"
            # In main files, link with -legacy suffix
            return f"({module_file}#{model_name}-legacy)"
        # Fallback for unmapped models (stay in same file)
        if for_legacy_file:
            return f"(#{model_name})"
        return f"(#{model_name}-legacy)"

    # Transform links from (#modelname) to proper cross-module references
    text = re.sub(r"\(#([a-z]+)\)", replace_link, text)

    # Transform workspace command references
    if for_legacy_file:
        # In legacy files, point to legacy workspace file
        text = re.sub(
            r"\(#anyscale-workspace_v2-([a-z]+)\)",
            r"(workspaces.md#anyscale-workspace_v2-\1)",
            text,
        )
    else:
        # In main files, point to main workspace file
        text = re.sub(
            r"\(#anyscale-workspace_v2-([a-z]+)\)",
            r"(workspaces.md#anyscale-workspace_v2-\1)",
            text,
        )

    return text


class LegacySDK:
    def __init__(self, name: str, docstring: str):
        self.name = name
        self.docstring = docstring

    @classmethod
    def from_md(cls, md: str) -> "LegacySDK":
        """
        Convert a blob of markdown into a LegacySDK object.
        """
        name = ""
        docstring = ""

        for line in md.split("\n"):
            if line.startswith("### "):
                name = line[4:]
            else:
                # Transform ./models.md links and local links
                transformed = _transform_legacy_links(
                    re.sub(r"\(./models\.md#([a-z]+)\)", r"(#\1-legacy)", line),
                    for_legacy_file=False,
                )
                docstring += transformed + "\n"

        return cls(name=name, docstring=docstring.strip())


class LegacyModel:
    def __init__(self, name: str, docstring: str):
        self.name = name
        self.docstring = docstring

    @classmethod
    def from_md(cls, md: str) -> "LegacyModel":
        """
        Convert a blob of markdown into a LegacyModel object.
        """
        name = ""
        docstring = ""

        for line in md.split("\n"):
            if line.startswith("## "):
                name = line[3:]
            else:
                docstring += _transform_legacy_links(line, for_legacy_file=False) + "\n"

        return cls(name=name, docstring=docstring.strip())


def _chunk(md_file: str, start: str, ends: List[str]) -> List[str]:
    """
    Split a markdown file into chunks based on the header.
    """
    chunks = []
    with open(md_file) as f:
        line = f.readline()
        while line:
            if not line.startswith(start):
                line = f.readline()
                continue
            chunk = line
            line = f.readline()
            while line and not any(line.startswith(end) for end in ends):
                chunk += line
                line = f.readline()

            chunks.append(chunk)

    return chunks


def parse_legacy_sdks(
    api_md_file: str, model_md_file: str
) -> Tuple[List[LegacySDK], List[LegacyModel]]:
    """
    Parse the legacy SDK markdown files into a list of LegacySDK objects.
    """
    legacy_sdks = [
        LegacySDK.from_md(chunk)
        for chunk in _chunk(api_md_file, "### ", ["### ", "## ", "# "])
    ]
    legacy_models = [
        LegacyModel.from_md(chunk)
        for chunk in _chunk(model_md_file, "## ", ["## ", "# "])
    ]

    return legacy_sdks, legacy_models
