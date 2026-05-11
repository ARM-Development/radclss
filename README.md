# RadCLss

**Extracted Radar Columns and In-Situ Sensors**

`radclss` extracts vertical radar columns over user-specified site locations and merges them with co-located in-situ sensor data (sondes, met stations, disdrometers, pluviometers, etc.) into a single xarray dataset / netCDF file. It supports multiple radar systems (e.g. CSAPR2, KASACR, XSACR, NEXRAD) and parallel processing via Dask.

## Installation

From PyPI:

```bash
pip install radclss
```

From source:

```bash
git clone https://github.com/ARM-Development/radclss.git
cd radclss
pip install -e .
```

## Quick Start

```python
import radclss

volumes = {
    "date": "20250520",
    "radar_csapr2cmac": [...],   # list of radar files
    "sonde": [...],
    "met_M1": [...],
    # ... additional radar / in-situ inputs
}

input_site_dict = {
    "M1": (34.34525, -87.33842, 293),   # (lat, lon, alt_m)
    "S4": (34.46451, -87.23598, 197),
}

columns = radclss.core.radclss(
    volumes,
    input_site_dict,
    "radar_csapr2cmac",
    serial=False,
    verbose=True,
    nexrad=True,
)

radclss.io.write_radclss_output(columns, "radclss_example.nc", "radclss.c2")

fig, ax = radclss.vis.create_radclss_columns("radclss_example.nc")
```

See `examples/bnf_example.py` for a full end-to-end script using the BNF (Bankhead National Forest) site, including Dask `LocalCluster` setup and multi-radar / multi-instrument inputs.

## Package Layout

- `radclss.core` — column extraction (`radclss.core.radclss`)
- `radclss.io` — netCDF output (`radclss.io.write_radclss_output`)
- `radclss.vis` — quicklook plots (`radclss.vis.create_radclss_columns`)
- `radclss.config` — default and output configuration
- `radclss.util` — column-processing utilities

## Development

Run tests and pre-commit hooks before committing:

```bash
pytest tests/
pre-commit run --all-files
```

## License

MIT — see `LICENSE`.
