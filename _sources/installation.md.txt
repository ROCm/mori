# Install MORI

Use a ROCm environment with Python 3.10 or later and a matching ROCm build of
PyTorch for tensor-based examples. Choose either the stable or nightly package;
both provide the `mori` import and must not be installed together.

The installation instructions below are included from the repository README so
package choices, system dependencies and fabric checks stay consistent.

```{include} ../README.md
---
start-after: '<!-- mori-installation-start -->'
end-before: '## Testing'
---
```

## Optional FlyDSL integration

The FlyDSL device API is optional. To install it with the stable MORI package:

```bash
pip install 'amd_mori[flydsl]'
```

The optional dependency is not version-pinned by MORI. When using MORI in a
framework image, preserve that image's tested FlyDSL version rather than
upgrading the compiler independently. See [EP backends](ep_backends.rst) for
paths that require FlyDSL and paths that can run with HIP alone.

## Next steps

* [Quickstart](quickstart.rst) provides runnable checks and API sketches.
* [EP backends](ep_backends.rst) explains API and backend selection.
* [CCO guide](MORI-CCO-GUIDE.md) describes communicator setup and transports.
