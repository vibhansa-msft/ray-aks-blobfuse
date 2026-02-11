# Anyscale Client

This directory contains a client wrapper that should be used for all communication with the Anyscale HTTP REST API.
The purpose of centralizing this logic is to:

- Avoid duplicating code.
- Keep all external dependencies in one place.
- Enable writing comprehensive unit tests for upstream components (without using mocks!) using the `FakeAnyscaleClient`.

## Installation

To install the anyscale client package in development mode:

```bash
cd frontend/cli
pip install -e .
```

This will install the `anyscale` package in editable mode, allowing you to make changes to the code and have them immediately available.

## Testing

The `AnyscaleClient` is tested using a fake version of the internal and external OpenAPI clients.

Upstream components should use the `FakeAnyscaleClient` to write their tests.
This client should mirror the behavior of the real `AnyscaleClient` as closely as possible (avoid making methods and functionality
complete "dummies").

### Running Tests

1. **Unit Tests**: Run the test suite to verify client functionality
   ```bash
   cd frontend/cli
   python -m pytest tests/unit/test_anyscale_client.py
   ```

2. **Integration Tests**: Test the client against the actual Anyscale API
   ```bash
   # Set up your Anyscale credentials
   export ANYSCALE_TOKEN="your_token_here"

   # Run integration tests
   python -m pytest tests/test_integrations.py
   ```

3. **All Client-Related Tests**: Run all tests that use the anyscale client
   ```bash
   cd frontend/cli
   python -m pytest tests/ -k "anyscale_client" -v
   ```

### Testing Job Submission

To test job submission functionality:

1. **Create a test job configuration file** (`job.yaml`):
   ```yaml
   name: test-job
   compute_config: default:1
   working_dir: /path/to/working/directory
   requirements:
   - numpy==1.24.0
   - pandas==2.0.0
   entrypoint: python your_script.py
   max_retries: 0
   ```

2. **Submit a test job**:
   ```bash
   anyscale job submit -f job.yaml
   ```

3. **Monitor job status**:
   ```bash
   anyscale job status <job_id>
   ```

4. **View job logs**:
   ```bash
   anyscale job logs <job_id>
   ```

### Example Test Job

Here's an example job configuration for testing:

```yaml
name: generate-doggos-embeddings
compute_config: doggos-azure:1
working_dir: abfss://cloud-dev-blob@anyscaleclouddev.dfs.core.windows.net/org_7c1Kalm9WcX2bNIjW53GUT/cld_wgmfc248s6t7513awyubirlwu9/runtime_env_packages/pkg_b60e2d10615fb9845a9bad7d9307547a.zip
requirements:
- matplotlib==3.10.0
- torch==2.7.1
- transformers==4.52.3
- scikit-learn==1.6.0
- mlflow==2.19.0
- ipywidgets==8.1.3
entrypoint: python doggos/embed.py
max_retries: 0
```

### Testing with Fake Client

For unit testing components that depend on the Anyscale client:

```python
from anyscale._private.anyscale_client import FakeAnyscaleClient

# Create a fake client for testing
fake_client = FakeAnyscaleClient()

# Use the fake client in your tests
# The fake client should behave like the real client
result = fake_client.submit_job(job_config)
assert result.job_id is not None
```

### Debugging

- Use `--verbose` flag for detailed output: `anyscale job submit -f job.yaml --verbose`
- Check job status in the UI: The CLI will provide a URL to view the job in the Anyscale console
- Use `--wait` flag to wait for job completion and stream logs: `anyscale job submit -f job.yaml --wait`

### Common Issues

1. **Authentication**: Ensure your Anyscale token is properly set
2. **Network**: Check your internet connection and firewall settings
3. **Dependencies**: Verify all required packages are installed
4. **Job Configuration**: Ensure your YAML file is properly formatted

For more detailed testing scenarios, refer to the test files in the `tests/` directory.
