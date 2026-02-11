Tool to autogenerate documentation for the Anyscale SDK/CLI that can be embedded in a [Docusarus](https://docusaurus.io/) site. If you're adding a new API reference in anyscale/product, follow this guide to see what changes you need to make: https://www.notion.so/anyscale-hq/How-to-document-new-anyscale-API-references-173027c809cb8080b8e3e53fc244d70b.

This is used by: https://github.com/anyscale/docs/tree/master/docs/reference.

Usage:
```bash
Usage: python -m anyscale._private.docgen [OPTIONS] OUTPUT_DIR

  Generate markdown docs for the Anyscale CLI & SDK.

Options:
  -r, --remove-existing  If set, all files in the 'output_dir' that were not
                         generated will be removed.
  --help                 Show this message and exit.
```
